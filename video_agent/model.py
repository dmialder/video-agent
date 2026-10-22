"""One local Qwen3-VL instance for tool decisions, inspection and answers."""
from pathlib import Path
import json
import time

class LocalVLM:
    def __init__(self,path: Path):
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        torch.manual_seed(17)
        self.torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation='sdpa')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()
        # The checkpoint's sampling defaults are irrelevant to greedy decoding.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.processor = AutoProcessor.from_pretrained(str(path),local_files_only=True, use_fast=False)
        self.calls = []

    def generate(self,system,prompt,images=None,max_tokens=430,tools=None,conversation=None):
        images = images or []
        content = [{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]
        messages = conversation or [{'role':'system','content':system},{'role':'user','content':content}]
        text = self.processor.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True)
        inputs = self.processor(text=[text],images=images or None,return_tensors='pt',padding=True)
        inputs = inputs.to(self.device)
        n = inputs['input_ids'].shape[1]
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False)
        answer = self.processor.batch_decode(outputs[:,n:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        self.calls.append({'system':system,'prompt':prompt,'conversation':messages if conversation else None,'tools_schema':tools,'images':len(images),'raw_output':answer,
                           'input_tokens':n,'output_tokens':outputs.shape[1]-n,'seconds':round(time.perf_counter()-started,3)})
        return answer

def json_object(raw):
    decoder = json.JSONDecoder()
    for i,c in enumerate(raw):
        if c=='{':
            try:
                result,_ = decoder.raw_decode(raw[i:])
                if isinstance(result,dict):
                    return result
            except json.JSONDecodeError:
                pass
    raise ValueError('Model output contains no valid JSON object')
