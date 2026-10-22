from pathlib import Path
import argparse
import json
import os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

def main():
    parser=argparse.ArgumentParser(description='Local tool-using video agent')
    parser.add_argument('command',choices=['ask','evaluate','demo','index'])
    parser.add_argument('question',nargs='?')
    parser.add_argument('--only',nargs='*')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if args.command=='index':
        from video_agent.index import CaptionIndex
        index=CaptionIndex(ROOT/'data/captions.vtt')
        (ROOT/'data/captions.json').write_text(json.dumps(index.export(),indent=2),encoding='utf-8')
        print(len(index.cues),'cues;',len(index.chunks),'search chunks;',index.duration,'seconds caption span')
        return
    from video_agent.model import LocalVLM
    from video_agent.agent import Agent,VideoTools
    print('Loading local VLM...',flush=True)
    model=LocalVLM(ROOT/'data/model')
    tools=VideoTools(ROOT,model)
    # Preserve the original 16-case experiment. The demo/ask path includes the
    # later, separately evaluated deterministic timecode conversion.
    agent=Agent(tools,model,time_hints=args.command!='evaluate')
    print(f'Ready on {model.device}; video {tools.duration:.2f} seconds.',flush=True)
    if args.command=='ask':
        if not args.question:
            parser.error('ask requires a question')
        result=agent.run(args.question)
        print(json.dumps({k:v for k,v in result.items() if k not in ('model_calls',)},ensure_ascii=False,indent=2))
    elif args.command=='evaluate':
        from video_agent.evaluation import evaluate
        evaluate(agent,ROOT,args.only)
    elif args.command=='demo':
        from video_agent.web import serve
        serve(agent,ROOT,args.port)

if __name__=='__main__':
    main()
