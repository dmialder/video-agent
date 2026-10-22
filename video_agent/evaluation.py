"""Transparent diagnostics: temporal evidence overlap and answer keyword checks."""
import csv
import json
from pathlib import Path

def interval_iou(a,b):
    intersection = max(0,min(a[1],b[1])-max(a[0],b[0]))
    union = max(a[1],b[1])-min(a[0],b[0])
    return intersection/union if union else 0.

def score(question,result):
    gold = question['gold_intervals']
    predicted = [[e['start'],e['end']] for e in result['evidence']]
    groups = question['answer_groups']
    answer = result['answer'].lower()
    # These are diagnostics, not a semantic judge; manual_review.csv remains separate.
    keyword = sum(any(w in answer for w in group) for group in groups)/len(groups) if groups else None
    hit = any(interval_iou(p,g)>0 for p in predicted for g in gold) if gold else None
    iou = max((interval_iou(p,g) for p in predicted for g in gold),default=0.) if gold else None
    return {'id':question['id'],'category':question['category'],'abstain':result['abstain'],
        'temporal_hit':hit,'best_temporal_iou':round(iou,4) if iou is not None else None,
        'keyword_coverage_heuristic':keyword,'correct_abstention':result['abstain'] if not gold else None,
        'tool_calls':result['tool_calls'],'seconds':result['seconds'],'stop_reason':result['stop_reason'],
        'answer':result['answer'],'run_dir':result['run_dir']}

def evaluate(agent,root:Path,only=None):
    questions = json.loads((root/'data/questions.json').read_text(encoding='utf-8'))['questions']
    folder = root/'results/evaluation'
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for q in questions:
        if only and q['id'] not in only:
            continue
        saved = folder/q['id']/'result.json'
        if saved.exists():
            result = json.loads(saved.read_text(encoding='utf-8'))
        else:
            print(q['id'],q['question'],flush=True)
            result = agent.run(q['question'],folder/q['id'])
        row = score(q,result)
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        (folder/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    if rows:
        with (folder/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    return rows
