"""Download the public, CC BY-NC-SA MIT example and its official captions."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import urllib.request
from urllib.parse import urljoin

try:
    import truststore
    truststore.inject_into_ssl()  # System trust store, with certificate verification intact.
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
PAGE = 'https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-1-algorithms-and-computation/'
VIDEO = 'https://archive.org/download/MIT6.006S20/MIT6_006S20_02_04_Lecture_1_300k.mp4'

def download(url, path):
    if path.exists() and path.stat().st_size > 0:
        return
    print('Downloading', url, flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Skoltech-Video-Agent-Educational-Project/1.0'})
    with urllib.request.urlopen(req, timeout=120) as response, path.with_suffix(path.suffix+'.part').open('wb') as out:
        while chunk := response.read(1024*1024):
            out.write(chunk)
    path.with_suffix(path.suffix+'.part').replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--discover', action='store_true')
    args = parser.parse_args()
    data = ROOT / 'data'
    data.mkdir(parents=True, exist_ok=True)
    download(PAGE, data/'source_page.html')
    source = (data/'source_page.html').read_text(encoding='utf-8')
    candidates = sorted(set(html.unescape(s) for s in re.findall(r'[^\s"<>]+\.(?:vtt|srt)(?:\?[^\s"<>]*)?', source)))
    print(json.dumps(candidates, indent=2), flush=True)
    if args.discover:
        return
    if not candidates:
        raise RuntimeError('Official captions URL missing; inspect source_page.html instead of inventing captions.')
    subtitle_url = urljoin(PAGE, candidates[0])
    subtitle_path = data/('captions'+('.vtt' if '.vtt' in subtitle_url else '.srt'))
    download(subtitle_url, subtitle_path)
    download(VIDEO, data/'lecture.mp4')
    manifest = {
        'title': '1. Algorithms and Computation', 'speaker': 'Jason Ku',
        'course': 'MIT 6.006, Spring 2020',
        'source_page': PAGE, 'youtube': 'https://www.youtube.com/watch?v=ZA-tUyM_y7s',
        'video_download': VIDEO, 'captions_download': subtitle_url,
        'license': 'CC BY-NC-SA 4.0', 'license_url': 'https://ocw.mit.edu/pages/privacy-and-terms-of-use/',
        'caption_method': 'Official MIT OCW captions; no speech recognition run.',
        'sha256': {p.name: hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (data/'lecture.mp4',subtitle_path)},
    }
    (data/'source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__ == '__main__':
    main()
