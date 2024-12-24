import re
import logging
import torch
import torch.nn as nn

from llm_reasoning.llm.base import LLM, InferenceConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an intelligent agent navigating a reasoning graph to solve a task. The graph consists of nodes, each representing a reasoning step. You can perform one of the following actions at each step:

1. **Continue**: Expand the current node by exploring deeper into its subnodes.
2. **Branch**: Explore a sibling node of the current node to investigate alternative reasoning paths.
3. **Backtrack**: Return to a previously visited node that may be suboptimal and to explore its alternative sibling node.

Note these sibling nodes are not given and will be dynamically generated once you choose to explore them.
""".strip()

USER_PROMPT = """
### Current Reasoning Path:
{reasoning_path}

### Output Format
- If you decide to continue explore the current reasoning path, please output "Continue".
- If you decide to explore an alternative reasoning path, please output "Branch".
- If you want to backtrack to a previous node, please output the node id as "Node Id: __".

You may optionally output explanations, but please put your final answer within <answer> and </answer> tags.
""".strip()

def collate_fn(batch):
    pass

class LLMPolicy(nn.Module):
    def __init__(self, model: LLM, inference_config: InferenceConfig):
        super().__init__()
        self.model = model
        self.inference_config = inference_config
    
    def _parse_action(self, action_text, reasoning_path):
        match = re.match(r"<answer>(.*?)</answer>", action_text, re.DOTALL)
        if match:
            action = match.group(1)
        else:
            logger.warning(f"Not parsable llm response: {action_text}")
            action = "Continue" # fallback to default action
            
        if action == "Continue":
            return 0
        elif action == "Branch":
            return 1
        elif re.search(r"Node Id:\s*(\d+)", action):
            back_node_id = re.search(r"Node Id:\s*(\d+)", action).group(1)
            cur_node_depth = reasoning_path[-1]["depth"]
            back_node_depth = {step["node_id"]: step["depth"] for step in reasoning_path}[back_node_id]
            backtrack_step = cur_node_depth - back_node_depth
            return backtrack_step + 1 # backtrack action id
        else:
            logger.warning(f"Not parsable action: {action}")
            return 0 # fallback to default action
    
    def _act(self, reasoning_path: list[dict]):
        formatted_reasoning_path = "\n".join(f"Node Id: {step['node_id']} Text: {step['text']}" for step in reasoning_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(reasoning_path=formatted_reasoning_path)}
        ]
        
        llm_response = self.model.call(messages, inference_config=self.inference_config)
        action = self._parse_action(llm_response.text, reasoning_path)
        
        return action
    
    def forward(self, ):
        pass