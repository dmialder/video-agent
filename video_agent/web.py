"""Loopback-only demo; no account, external API, or exposed project directory."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import re
import threading
import uuid
from urllib.parse import urlsplit,unquote

def serve(agent,root:Path,port=8765):
    jobs={}
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,data,status=200):
            raw=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

        def do_GET(self):
            path=unquote(urlsplit(self.path).path)
            if path=='/api/source':
                return self.send_json({**json.loads((root/'data/source.json').read_text(encoding='utf-8')),'duration':agent.tools.duration,'model':'Qwen3-VL-2B-Instruct (local)'})
            if path=='/api/examples':
                examples=[]
                for file in sorted((root/'results/evaluation').glob('q*/result.json')):
                    result=json.loads(file.read_text(encoding='utf-8'))
                    examples.append({'id':file.parent.name,'label':'Исходный прогон '+file.parent.name,'question':result['question']})
                for file in sorted((root/'results/timecode_ablation').glob('q*/result.json')):
                    result=json.loads(file.read_text(encoding='utf-8'))
                    examples.append({'id':'time_'+file.parent.name,'label':'С пересчётом времени '+file.parent.name,'question':result['question']})
                return self.send_json(examples)
            if path.startswith('/api/example/'):
                key=path.rsplit('/',1)[-1]
                if not re.fullmatch(r'(?:time_)?q\d{2}',key):
                    return self.send_json({'error':'Invalid example'},400)
                file=root/('results/timecode_ablation' if key.startswith('time_') else 'results/evaluation')/key.removeprefix('time_')/'result.json'
                return self.send_json(json.loads(file.read_text(encoding='utf-8'))) if file.exists() else self.send_json({'error':'Not found'},404)
            if path.startswith('/api/job/'):
                return self.send_json(jobs.get(path.rsplit('/',1)[-1],{'status':'not_found'}))
            if path=='/':
                return self.send_file(root/'web/index.html')
            if path=='/video':
                return self.send_file(root/'data/lecture.mp4')
            if path.startswith('/results/'):
                file=(root/path.lstrip('/')).resolve()
                if file.is_relative_to((root/'results').resolve()) and file.suffix=='.jpg':
                    return self.send_file(file)
            self.send_error(404)

        def send_file(self,file):
            if not file.is_file():
                return self.send_error(404)
            size=file.stat().st_size
            start,end=0,size-1
            header=self.headers.get('Range')
            if header:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
                if not match:
                    return self.send_error(416)
                start=int(match[1]);end=min(int(match[2]) if match[2] else size-1,size-1)
                if start>end or start>=size:
                    return self.send_error(416)
            self.send_response(206 if header else 200)
            self.send_header('Content-Type',mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
            if header:
                self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with file.open('rb') as stream:
                    stream.seek(start);remaining=end-start+1
                    while remaining:
                        chunk=stream.read(min(1024*1024,remaining))
                        if not chunk:break
                        self.wfile.write(chunk);remaining-=len(chunk)
            except (ConnectionResetError,BrokenPipeError):
                pass

        def do_POST(self):
            if urlsplit(self.path).path!='/api/ask':
                return self.send_json({'error':'Not found'},404)
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                return self.send_json({'error':'Origin rejected'},403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8192:raise ValueError('Invalid request size')
                question=json.loads(self.rfile.read(length))['question']
                if not isinstance(question,str) or not 1<=len(question.strip())<=1000:raise ValueError('Question: 1..1000 characters')
            except (ValueError,KeyError) as exc:
                return self.send_json({'error':str(exc)},400)
            if not lock.acquire(blocking=False):
                return self.send_json({'error':'An inference is already running. Please wait.'},409)
            key=uuid.uuid4().hex
            jobs[key]={'status':'running'}
            def work():
                try:jobs[key]={'status':'done','result':agent.run(question)}
                except Exception as exc:jobs[key]={'status':'error','error':str(exc)}
                finally:lock.release()
            threading.Thread(target=work,daemon=True).start()
            return self.send_json({'job':key},202)
    print(f'Open http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
