"""Fetch a public model revision; never load remote Python code."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_XET'] = '1'
import truststore
truststore.inject_into_ssl()
import json,re
from huggingface_hub import HfApi, snapshot_download

model_id = 'Qwen/Qwen3-VL-2B-Instruct'
pin_path=ROOT/'data/model_revision.json'
pin=json.loads(pin_path.read_text(encoding='utf-8-sig'))['revision'] if pin_path.exists() else None
if pin is not None and not re.fullmatch(r'[0-9a-f]{40}',pin):
    raise ValueError('Invalid pinned model revision')
info = HfApi(token=False).model_info(model_id,revision=pin)
revision = info.sha
print(model_id, revision, flush=True)
snapshot_download(model_id, revision=revision, token=False, local_dir=ROOT/'data/model',
                  allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE*','README.md'], max_workers=2)
(ROOT/'data/model_revision.json').write_text(json.dumps({'model_id':model_id,'revision':revision,'license':'Apache-2.0'},indent=2),encoding='utf-8')
print('Model downloaded.',flush=True)
