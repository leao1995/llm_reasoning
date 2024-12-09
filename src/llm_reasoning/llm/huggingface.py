import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from llm_reasoning.llm.base import LLM, LLMResponse, InferenceConfig

class HuggingFaceModel(LLM):
    model_name: str
    batch_size: int
    device: str
    
    _model: AutoModelForCausalLM
    _tokenizer: AutoTokenizer
    
    def __init__(self, **data):
        super().__init__(**data)
        
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def batch_call(self, batch_messages: list[list[dict]], inference_config: InferenceConfig) -> list[LLMResponse]:
        chat_prompts = [
            self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False if messages[-1]["role"] == "assistant" else True,
                continue_final_message=True if messages[-1]["role"] == "assistant" else False,
            )
            for messages in batch_messages
        ]
        
        responses = []
        for start in tqdm(range(0, len(chat_prompts), self.batch_size), desc="llm calls"):
            batch_prompts = chat_prompts[start:start+self.batch_size]
            tokenized_inputs = self._tokenizer(
                batch_prompts, 
                padding=True, 
                truncation=True, 
                add_special_tokens=False, # chat_template already added special tokens
                return_tensors="pt"
            ).to(self.device)
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
                
                tokens = []
                logprobs = []
                confidences = []
                for step_logits, token_id in zip(outputs.logits, generated_token_ids):
                    if token_id == self._tokenizer.pad_token_id:
                        break
                    scores = torch.nn.functional.log_softmax(step_logits[i], dim=-1)
                    logprob = scores[token_id]
                    confidence = torch.exp(logprob-torch.logsumexp(scores.topk(5)[0], dim=-1))
                    tokens.append(self._tokenizer.decode(token_id, skip_special_tokens=True))
                    logprobs.append(logprob.item())
                    confidences.append(confidence.item())
                    
                token_probs = [{"token": t, "logprob": p, "confidence": c} for t, p, c in zip(tokens, logprobs, confidences)]
                text, token_probs, last_token_idx = truncate_generations(text, token_probs, inference_config.stop_sequences)
                logprobs = [token["logprob"] for token in token_probs]
                confidences = [token["confidence"] for token in token_probs]
                    
                perplexity = torch.exp(-torch.tensor(logprobs).mean()).item()
                # The hidden state is the one used to generate this token
                # therefore, for the first token, it's the hidden state of the inputs, with shape (batch_size, #input_tokens, dim)
                # for other tokens, it's of shape (batch_size, 1, dim)
                embedding = outputs.hidden_states[last_token_idx][-1][i][-1].data.cpu().float() # token index -> last layer -> batch_idx -> last token
                
                responses.append(
                    LLMResponse(
                        text=text,
                        logprobs=logprobs,
                        confidences=confidences,
                        perplexity=perplexity,
                        embedding=embedding,
                    )
                )
                    
        return responses
    
    def get_prompt_embedding(self, messages: list[dict]) -> torch.Tensor:
        assert messages[-1]["role"] == "user"
        
        chat_prompt = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False,
        )
        tokenized_inputs = self._tokenizer(
            chat_prompt, 
            padding=True, 
            truncation=True, 
            add_special_tokens=False, # chat_template already added special tokens
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**tokenized_inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][0][-1].data.cpu().float() # last layer -> first batch id -> last token
        
        return embedding
    
def truncate_generations(text: str, token_probs: list[dict], stop_sequences: list[str]):
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
    assert text == ''.join(token['token'] for token in token_probs)
    
    # Strip leading whitespace from text
    stripped_text = text.lstrip()
    leading_whitespace_len = len(text) - len(stripped_text)
    
    # Adjust token_start_index by summing actual token contributions in output text
    token_start_index = 0
    cumulative_length = 0
    for i, token in enumerate(token_probs):
        # Extract the token text as it appears in the output
        token_length = len(token.get("token", ""))
        
        # Skip tokens contributing to the leading whitespace
        cumulative_length += token_length
        if cumulative_length > leading_whitespace_len:
            token_start_index = i
            break
            
    # Update the text and tokens after removing leading whitespace
    text = stripped_text
    token_probs = token_probs[token_start_index:]
    
    last_token_idx = len(token_probs) - 1  # To store the index of the last token after truncation
    
    for stop_seq in stop_sequences:
        # Skip leading stop sequences
        index = 0
        while text.startswith(stop_seq, index):
            index += len(stop_seq)
            
        # Find the next occurrence of the stop sequence
        stop_index = text.find(stop_seq, index)
        if stop_index != -1:
            # Truncate the text at the first valid stop sequence
            text = text[:stop_index]
            
            # Calculate the corresponding token length
            cumulative_length = 0
            truncated_token_length = 0
            for i, token in enumerate(token_probs):
                token_length = len(token.get("token", ""))
                token_end = cumulative_length + token_length
                
                # Check if the stop sequence starts within or ends exactly at the token's end
                if cumulative_length < stop_index <= token_end:
                    truncated_token_length = i + 1  # Include this token
                    last_token_idx = token_start_index + i
                    break
                
                cumulative_length = token_end
            
            # Truncate token probabilities
            token_probs = token_probs[:truncated_token_length]
    
    # Return the original text and token probabilities if no stop sequence is found
    return text, token_probs, last_token_idx