"""Timestamped captions and a small deterministic BM25 index."""
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import html
import math
import re

STOP = set('a an the of in on at to for from and or is are was were be it that this what when where why how does do did can you i we he she they with about as by which some his her its'.split())

def tokens(text):
    words = re.findall(r'[a-z0-9]+',text.lower())
    # Light inflection normalization, used identically for documents and queries.
    return [w[:-1] if len(w)>4 and w.endswith('s') else w for w in words if w not in STOP]

def seconds(s):
    parts = [float(p) for p in s.replace(',','.').split(':')]
    return sum(v*60**i for i,v in enumerate(reversed(parts)))

def clock(t):
    t = max(0,int(t))
    return f'{t//3600:02d}:{t//60%60:02d}:{t%60:02d}'

def explicit_timecodes(text):
    result=[]
    for match in re.finditer(r'(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])',text):
        hour,minute,second=match.groups()
        if int(minute)>=60 or int(second)>=60:
            continue
        result.append({'timecode':match.group(0),'seconds':int(hour or 0)*3600+int(minute)*60+int(second)})
    return result

@dataclass
class Cue:
    start: float
    end: float
    text: str

class CaptionIndex:
    def __init__(self, path: Path, chunk_seconds=35):
        raw = path.read_text(encoding='utf-8-sig').replace('\r','')
        self.cues = []
        for block in re.split(r'\n\s*\n',raw):
            lines = block.strip().splitlines()
            times = next((i for i,s in enumerate(lines) if '-->' in s),None)
            if times is None:
                continue
            left,right = lines[times].split('-->')
            text = html.unescape(re.sub(r'<[^>]*>','', ' '.join(lines[times+1:])))
            if text.strip():
                self.cues.append(Cue(seconds(left.strip()),seconds(right.strip().split()[0]),text.strip()))
        if not self.cues:
            raise ValueError('No valid caption cues')
        self.cues.sort(key=lambda c:c.start)
        self.duration = self.cues[-1].end
        self.chunks = []
        for start in range(0,math.ceil(self.duration),chunk_seconds):
            cues = self.between(start,start+chunk_seconds)
            if cues:
                self.chunks.append({'id':f'S{len(self.chunks):03d}', 'start':cues[0].start,'end':cues[-1].end,'text':' '.join(c.text for c in cues)})
        self.counts = [Counter(tokens(c['text'])) for c in self.chunks]
        self.df = Counter(w for count in self.counts for w in count)
        self.avglen = sum(map(lambda c:sum(c.values()), self.counts))/len(self.counts)

    def between(self,start,end):
        return [c for c in self.cues if c.start < end and c.end > start]

    def search(self,query,k=3):
        q = set(tokens(query))
        if not q:
            return []
        scores = []
        for i,counts in enumerate(self.counts):
            score = 0.
            length = sum(counts.values())
            for word in q:
                f = counts[word]
                if f:
                    idf = math.log(1+(len(self.chunks)-self.df[word]+.5)/(self.df[word]+.5))
                    score += idf*f*2.5/(f+1.5*(.25+.75*length/self.avglen))
            if score>0:
                scores.append((score,i))
        scores.sort(key=lambda p:(-p[0],p[1]))
        return [{**self.chunks[i],'score':round(score,3)} for score,i in scores[:k]]

    def export(self):
        return [asdict(c) for c in self.cues]
