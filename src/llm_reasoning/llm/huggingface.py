import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

logger = logging.getLogger(__name__)

class HuggingFaceModel(LLM):
    model_name: str
    batch_size: int
    quantize: bool
    
    _model: AutoModelForCausalLM
    _tokenizer: AutoTokenizer
    
    def __init__(self, **data):
        super().__init__(**data)
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_in_4bit=self.quantize,
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._model.generation_config.pad_token_id = self._tokenizer.eos_token_id
    
    def batch_call(self, batch_messages: list[list[dict]], inference_config: InferenceConfig) -> list[LLMResponse]:
        chat_prompts = [
            self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False if messages[-1]["role"] == "assistant" else True,
                continue_final_message=True if messages[-1]["role"] == "assistant" else False,
                chat_template=inference_config.chat_template, # custom chat template
            )
            for messages in batch_messages
        ]
        
        responses = []
        for start in range(0, len(chat_prompts), self.batch_size):
            batch_prompts = chat_prompts[start:start+self.batch_size]
            tokenized_inputs = self._tokenizer(
                batch_prompts, 
                padding=True, 
                truncation=True, 
                add_special_tokens=False, # chat_template already added special tokens
                return_tensors="pt"
            ).to(self._model.device)
            num_input_tokens = tokenized_inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **tokenized_inputs,
                    do_sample=True if inference_config.temperature > 0 else False,
                    temperature=inference_config.temperature,
                    top_p=inference_config.top_p,
                    max_new_tokens=inference_config.max_tokens,
                    return_dict_in_generate=True,
                    output_logits=True,
                    output_hidden_states=True,
                )
                
            for i in range(len(batch_prompts)):
                generated_token_ids = outputs.sequences[i][num_input_tokens:]
                text = self._tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                finish_reason = "stop" if self._tokenizer.eos_token_id in generated_token_ids else "length"
                
                tokens = []
                logprobs = []
                confidences = []
                for step_logits, token_id in zip(outputs.logits, generated_token_ids):
                    if token_id in [self._tokenizer.pad_token_id, self._tokenizer.eos_token_id] + self._model.generation_config.eos_token_id:
                        break
                    scores = torch.nn.functional.log_softmax(step_logits[i], dim=-1)
                    logprob = scores[token_id]
                    confidence = torch.exp(logprob-torch.logsumexp(scores.topk(5)[0], dim=-1))
                    tokens.append(self._tokenizer.decode(token_id, skip_special_tokens=True))
                    logprobs.append(logprob.item())
                    confidences.append(confidence.item())
                    
                token_probs = [{"token": t, "logprob": p, "confidence": c} for t, p, c in zip(tokens, logprobs, confidences)]
                text, token_probs, finish_reason, last_token_idx = truncate_generations(text, token_probs, finish_reason, inference_config.stop_sequences)
                logprobs = [token["logprob"] for token in token_probs]
                confidences = [token["confidence"] for token in token_probs]
                    
                perplexity = torch.exp(-torch.tensor(logprobs).mean()).item()
                # The hidden state is the one used to generate this token
                # therefore, for the first token, it's the hidden state of the inputs, with shape (batch_size, #input_tokens, dim)
                # for other tokens, it's of shape (batch_size, 1, dim)
                embedding = outputs.hidden_states[last_token_idx][-1][i][-1].data.cpu().float() # generation index -> last layer -> batch_idx -> last token
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                
                responses.append(
                    LLMResponse(
                        text=text,
                        finish_reason=finish_reason,
                        logprobs=logprobs,
                        confidences=confidences,
                        perplexity=perplexity,
                        embedding=embedding,
                    )
                )
                    
        return responses
    
    def batch_inference(self, batch_input_ids: list[list[int]], inference_config: InferenceConfig):
        responses = []
        for start in range(0, len(batch_input_ids), self.batch_size):
            batch_inputs = self._tokenizer.pad(
                {"input_ids": batch_input_ids[start:start+self.batch_size]},
                return_tensors="pt"
            ).to(self._model.device)
            num_input_tokens = batch_inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **batch_inputs,
                    do_sample=True if inference_config.temperature > 0 else False,
                    temperature=inference_config.temperature,
                    top_p=inference_config.top_p,
                    max_new_tokens=inference_config.max_tokens,
                    return_dict_in_generate=True,
                    output_logits=True,
                    output_hidden_states=True,
                )
            
            for i in range(batch_inputs.input_ids.shape[0]):
                input_ids = batch_input_ids[start * self.batch_size + i]
                generated_token_ids = outputs.sequences[i][num_input_tokens:]
                text = self._tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                finish_reason = "stop" if self._tokenizer.eos_token_id in generated_token_ids else "length"
                
                tokens = []
                token_ids = []
                logprobs = []
                confidences = []
                for step_logits, token_id in zip(outputs.logits, generated_token_ids):
                    if token_id in [self._tokenizer.pad_token_id, self._tokenizer.eos_token_id] + self._model.generation_config.eos_token_id:
                        break
                    scores = torch.nn.functional.log_softmax(step_logits[i], dim=-1)
                    logprob = scores[token_id]
                    confidence = torch.exp(logprob-torch.logsumexp(scores.topk(5)[0], dim=-1))
                    tokens.append(self._tokenizer.decode(token_id, skip_special_tokens=True))
                    token_ids.append(token_id.item())
                    logprobs.append(logprob.item())
                    confidences.append(confidence.item())
                
                token_probs = [{"token": t, "token_id": e, "logprob": p, "confidence": c} for t, e, p, c in zip(tokens, token_ids, logprobs, confidences)]
                text, token_probs, finish_reason, last_token_idx = truncate_generations(text, token_probs, finish_reason, inference_config.stop_sequences)
                token_ids = input_ids + [token["token_id"] for token in token_probs]
                logprobs = [token["logprob"] for token in token_probs]
                confidences = [token["confidence"] for token in token_probs]
                
                perplexity = torch.exp(-torch.tensor(logprobs).mean()).item()
                # The hidden state is the one used to generate this token
                # therefore, for the first token, it's the hidden state of the inputs, with shape (batch_size, #input_tokens, dim)
                # for other tokens, it's of shape (batch_size, 1, dim)
                embedding = outputs.hidden_states[last_token_idx][-1][i][-1].data.cpu().float() # generation index -> last layer -> batch_idx -> last token
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                
                responses.append(
                    LLMResponse(
                        text=text,
                        token_ids=token_ids,
                        finish_reason=finish_reason,
                        logprobs=logprobs,
                        confidences=confidences,
                        perplexity=perplexity,
                        embedding=embedding,
                    )
                )
                
        return responses
    
    def encode(self, messages: list[dict], inference_config: InferenceConfig) -> list[int]:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False if messages[-1]["role"] == "assistant" else True,
            continue_final_message=True if messages[-1]["role"] == "assistant" else False,
            chat_template=inference_config.chat_template, # custom chat template
        )
    
    def get_prompt_embedding(self, messages: list[dict], inference_config: InferenceConfig) -> torch.Tensor:
        assert messages[-1]["role"] == "user"
        
        chat_prompt = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False,
            continue_final_message=True,
            chat_template=inference_config.chat_template,
            return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            outputs = self._model(chat_prompt, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][0][-1].data.cpu().float() # last layer -> first batch idx -> last token
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def get_answer_probs(self, messages: list[dict], answer_candidates: list[str], inference_config: InferenceConfig, normalize: bool=True):
        assert messages[-1]["role"] == "user"
        
        prefix = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            chat_template=inference_config.chat_template,
        )
        
        batch_messages = [messages + [{"role": "assistant", "content": cand}] for cand in answer_candidates]

        chat_prompts = [
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_genertion_prompt=False,
                continue_final_message=True,
                chat_template=inference_config.chat_template,
            )
            for messages in batch_messages
        ]
        tokenized_inputs = self._tokenizer(
            chat_prompts, 
            padding=True, 
            truncation=True, 
            add_special_tokens=False, # chat_template already added special tokens
            return_tensors="pt"
        ).to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model(**tokenized_inputs)
        
        log_probs = torch.nn.functional.log_softmax(outputs.logits[:, len(prefix)-1:-1], dim=-1)
        candidate_ids = tokenized_inputs.input_ids[:, len(prefix):]
        
        log_probs = log_probs.gather(2, candidate_ids.unsqueeze(-1)).squeeze(-1)
        pad_mask = (candidate_ids == self._tokenizer.pad_token_id) | (candidate_ids == self._tokenizer.eos_token_id)
        log_probs[pad_mask] = 0
        
        seq_log_probs = log_probs.sum(dim=1)
        
        if normalize:
            answer_probs = torch.exp(seq_log_probs - torch.logsumexp(seq_log_probs, dim=0))
            
            return answer_probs.data.cpu().float().tolist()
        
        return seq_log_probs.data.cpu().float().tolist()
    
def truncate_generations(text: str, token_probs: list[dict], finish_reason: str, stop_sequences: list[str]):
    '''
    Truncates the input text and corresponding token probabilities at the first occurrence of any stop sequence,
    skipping leading stop sequences to avoid empty generations.
    
    Parameters:
        text (str): The input string to truncate.
        token_probs (list[dict]): A list of token probability dictionaries, each corresponding to a token in the text.
        stop_sequences (list[str]): A list of stop sequences to consider.
    
    Returns:
        tuple: A tuple containing:
            - truncated_text (str): The truncated string.
            - truncated_token_probs (list[dict]): The truncated token probabilities.
    '''
    if text.strip() != ''.join(token['token'] for token in token_probs).strip():
        logger.warning(f"tokens does not match decoded text:\n---\n{text}\n---\n{''.join(token['token'] for token in token_probs)}\n---")
    
    last_token_idx = len(token_probs) - 1  # To store the index of the last token after truncation
    
    # Strip leading whitespace from text
    stripped_text = text.lstrip()
    leading_whitespace_len = len(text) - len(stripped_text)
    
    for stop_seq in stop_sequences:
        # Skip leading stop sequences
        index = leading_whitespace_len
        while text.startswith(stop_seq, index):
            index += len(stop_seq)
            
        # Find the next occurrence of the stop sequence
        stop_index = text.find(stop_seq, index)
        if stop_index != -1:
            # Truncate the text at the first valid stop sequence
            text = text[:stop_index]
            finish_reason = "truncation"
            
            # Calculate the corresponding token length
            cumulative_length = 0
            truncated_token_length = 0
            for i, token in enumerate(token_probs):
                token_length = len(token.get("token", ""))
                token_end = cumulative_length + token_length
                
                # Check if the stop sequence starts within or ends exactly at the token's end
                if cumulative_length < stop_index <= token_end:
                    truncated_token_length = i + 1  # Include this token
                    last_token_idx = i
                    break
                
                cumulative_length = token_end
            
            # Truncate token probabilities
            token_probs = token_probs[:truncated_token_length]
    
    # Return the original text and token probabilities if no stop sequence is found
    return text, token_probs, finish_reason, last_token_idx