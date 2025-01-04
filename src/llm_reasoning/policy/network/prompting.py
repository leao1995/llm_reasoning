import re
import logging
import torch
import torch.nn as nn
from retry import retry

from llm_reasoning.llm.base import LLM, InferenceConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an AI assistant designed to navigate through a reasoning graph to solve problems. At each node in the graph, you can take one of these actions:

1. Continue: Dive deeper from current node to explore its subnodes
2. Branch: Explore alternative parallel paths by investigating sibling nodes
3. Backtrack: Return to a previous node to explore different possibilities
4. Terminate: End the reasoning process when you've reached a satisfactory answer

Important notes:
- Sibling nodes are generated dynamically when you choose to explore them
- Your goal is to find the most logical path to the correct answer
- Consider each step carefully before choosing an action
- Please choose only from the available actions list given below
- Choose `Branch` when current node seem incorrect or inefficient towards a solution
- Choose `Backtrack` to revise a previous node that may lead to incorrect or inefficient solution
- When `Backtrack` to a previous node in the current path, specify its node id in the format 'Node Id: NODE_ID'
- Choose `Terminate` only when you're confident in the complete solution
""".strip()

USER_PROMPT = """
# QUESTION:
Here is the problem to be solved:
{question}

# CURRENT REASONING PATH:
Node IDs and reasoning steps:
{reasoning_path}

# AVAILABLE ACTIONS:
Due to graph constraints in depth and breadth, not all actions are available to choose at each step. The following actions are currently available:
{available_actions}

Instructions:
1. Analyze where you are in solving the problem
2. Review which actions are available to you
3. Decide your next action based on:
   - Is the current path promising?
   - Are there better approaches to try?
   - Have you reached a complete solution?
4. You can only choose one action from the available actions above 
5. Format your final action choice as: <answer>YOUR_CHOSEN_ACTION</answer> and optionally provide your thinking process. `YOUR_CHOSEN_ACTION` should be one of the available actions.
""".strip()

def collate_fn(batch):
    questions = []
    reasoning_paths = []
    for ex in batch:
        questions.append(ex["question"])
        reasoning_paths.append(ex["reasoning_path"])
        
    return {
        "questions": questions,
        "reasoning_paths": reasoning_paths,
    }

class ParsingException(Exception):
    pass

class LLMPolicy(nn.Module):
    def __init__(self, model: LLM, inference_config: InferenceConfig, num_actions: int):
        super().__init__()
        self.model = model
        self.inference_config = inference_config
        self.num_actions = num_actions
    
    def _parse_action(self, action_text, reasoning_path, available_actions):
        match = re.search(r"<answer>(.*?)</answer>", action_text, re.DOTALL)
        if match:
            action = match.group(1)
        else:
            raise ParsingException(f"answer tag does not match: {action_text}")
        
        if action not in available_actions:
            raise ParsingException(f"invalid actions: {action} not in {available_actions}")
            
        if action == "Continue":
            return 0
        elif action == "Branch":
            return 1
        elif "Node Id" in action:
            back_node_id = re.search(r"Node Id:\s*(\d+)", action).group(1)
            cur_node_depth = reasoning_path[-1]["depth"]
            path_node_depth = {step["node_id"]: step["depth"] for step in reasoning_path}
            back_node_depth = path_node_depth[int(back_node_id)]
            backtrack_step = cur_node_depth - back_node_depth
            return backtrack_step + 1 # backtrack action id
        elif action == "Terminate":
            return self.num_actions - 1
        else:
            raise ValueError()
    
    @retry(ParsingException, tries=5)
    def _act(self, question: str, reasoning_path: list[dict], available_actions: list[str]):
        formatted_reasoning_path = "\n".join(f"Node Id: {step['node_id']} Text: {step['text']}" for step in reasoning_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                question=question, 
                reasoning_path=formatted_reasoning_path,
                available_actions=available_actions,
            )}
        ]
        llm_response = self.model.batch_call([messages], inference_config=self.inference_config)[0]
        action = self._parse_action(llm_response.text, reasoning_path, available_actions)
        
        return action
    
    def _get_available_actions(self, reasoning_path, action_mask):
        available_actions = []
        
        actionid_to_nodeid = {i+1: node["node_id"] for i, node in enumerate(reasoning_path[::-1])}
        
        for i, m in enumerate(action_mask):
            if not m: continue
            
            if i == 0:
                available_actions.append("Continue")
            elif i == 1:
                available_actions.append("Branch")
            elif i == self.num_actions-1:
                available_actions.append("Terminate")
            else:
                available_actions.append(f"Node Id: {actionid_to_nodeid[i]}")
                
        return available_actions
    
    def forward(self, questions: list[str], reasoning_paths: list[list[dict]], action_masks: torch.Tensor):
        actions = []
        for i, (question, reasoning_path) in enumerate(zip(questions, reasoning_paths)):
            available_actions = self._get_available_actions(reasoning_path, action_masks[i])
            try:
                action = self._act(question, reasoning_path, available_actions)
            except ParsingException as e:
                logger.warning(f"Parsing error in LLMPolicy: {e}")
                action = 0 if 'Continue' in available_actions else self.num_actions-1 # fallback to COT
            actions.append(action)
        actions = torch.tensor(actions)
        
        return {
            'dist': None,
            'act': actions,
            'logp': torch.empty_like(actions),
            'vpred': torch.empty_like(actions),
        }
        