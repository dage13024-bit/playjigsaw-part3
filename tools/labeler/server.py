#!/usr/bin/env python3
"""
极简画框工具（浏览器版）——用来人工修正主体检测的框。

启动：
    .venv/bin/python tools/labeler/server.py

然后浏览器打开 http://127.0.0.1:8765

默认配置：
    图片：datasets/p3det_obj/images/val
    预标注：datasets/p3det_obj/labels/val（现有的弱框，会预加载）
    保存到：datasets/p3det_obj/labels_manual/val

操作：
    - 鼠标拖拽 = 画一个新框（每张图只有一个主体，所以覆盖旧框）
    - 下拉框 / 数字键 1-9 = 选类别
    - Enter = 保存并下一张；← → = 上一张/下一张
    - Delete = 清空当前框
    - 什么都不改直接 Enter = 接受预标注
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES: list[Path] = []
NAMES: list[str] = []

HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>拼图主体框标注</title>
<style>
  body{margin:0;background:#11151f;color:#e6e9f0;font:14px/1.5 -apple-system,"PingFang SC",sans-serif}
  .wrap{display:flex;gap:16px;padding:16px}
  .stage{flex:1;display:flex;align-items:center;justify-content:center;background:#0b0e15;border-radius:8px;min-height:720px}
  canvas{cursor:crosshair;image-rendering:auto}
  .panel{width:260px;display:flex;flex-direction:column;gap:12px}
  h1{font-size:16px;margin:0 0 4px}
  .muted{color:#8b93a7;font-size:12px}
  select,button{background:#1c2230;color:#e6e9f0;border:1px solid #2c3548;border-radius:6px;padding:8px;font-size:14px}
  button{cursor:pointer}
  button:hover{border-color:#4a5a7a}
  .row{display:flex;gap:8px}
  .row>*{flex:1}
  .progress{font-size:20px;font-weight:600}
  .kbd{background:#1c2230;border:1px solid #2c3548;border-radius:4px;padding:1px 5px;font-size:12px}
  .cls{display:flex;flex-wrap:wrap;gap:4px}
  .cls button{padding:4px 6px;font-size:12px}
  .cls button.on{background:#2563eb;border-color:#3b82f6}
</style>
</head>
<body>
<div class="wrap">
  <div class="stage"><canvas id="cv"></canvas></div>
  <div class="panel">
    <h1>主体框标注</h1>
    <div class="muted">拖拽画框；Enter 保存下一张；什么都不改直接 Enter = 接受预标注</div>
    <div class="progress"><span id="pos">-</span></div>
    <div class="muted" id="fname"></div>
    <select id="cls"></select>
    <div class="row">
      <button id="prev">← 上一张</button>
      <button id="next">保存 →</button>
    </div>
    <button id="del">清空当前框</button>
    <div class="muted">
      <span class="kbd">1-9</span> 选类别 ·
      <span class="kbd">Enter</span> 保存下一张 ·
      <span class="kbd">←/→</span> 翻页 ·
      <span class="kbd">Del</span> 清空
    </div>
    <div class="muted" id="status"></div>
  </div>
</div>
<script>
let data=null, idx=0, img=null, scale=1, boxes=[], drag=null, start=null, dirty=false;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const clsSel=document.getElementById('cls');

function fit(){
  const maxW=900, maxH=700;
  scale=Math.min(maxW/img.width, maxH/img.height, 4.0);
  cv.width=Math.round(img.width*scale); cv.height=Math.round(img.height*scale);
}
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if(img) ctx.drawImage(img,0,0,cv.width,cv.height);
  const box = drag || (boxes[0]||null);
  if(box){
    const [c,cx,cy,w,h]=box;
    const x=(cx-w/2)*img.width*scale, y=(cy-h/2)*img.height*scale;
    const bw=w*img.width*scale, bh=h*img.height*scale;
    ctx.strokeStyle='#22c55e'; ctx.lineWidth=2;
    ctx.strokeRect(x,y,bw,bh);
    ctx.fillStyle='#22c55e';
    ctx.fillRect(x,y-18,Math.max(70,bw),18);
    ctx.fillStyle='#04120a'; ctx.font='12px sans-serif';
    ctx.fillText(data.names[c]||('class '+c), x+4, y-5);
  }
}
function load(){
  drag=null; start=null;
  const it=data.images[idx];
  document.getElementById('pos').textContent=(idx+1)+' / '+data.images.length;
  document.getElementById('fname').textContent=it.name+(it.done?'（已保存）':'');
  boxes=it.boxes.map(b=>b.slice());
  img=new Image();
  img.onload=()=>{fit();draw();};
  img.src='/img/'+encodeURIComponent(it.name);
}
function save(thenNext){
  const it=data.images[idx];
  fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:it.name,boxes:boxes})}).then(r=>r.json()).then(j=>{
      if(j.ok){it.done=true;document.getElementById('status').textContent='已保存 '+it.name;
        if(thenNext){idx=(idx+1)%data.images.length; dirty=false; load();}}
      else document.getElementById('status').textContent='保存失败: '+j.error;
    });
}
function canvasXY(e){
  const r=cv.getBoundingClientRect();
  return [e.clientX-r.left, e.clientY-r.top];
}
cv.addEventListener('mousedown',e=>{
  const [x,y]=canvasXY(e);
  start=[x,y];                       // 起点单独存，不能被覆盖
  drag=[Number(clsSel.value), x/img.width/scale, y/img.height/scale, 0, 0];
  dirty=true; draw();
});
cv.addEventListener('mousemove',e=>{
  if(!drag || !start) return;
  const [x,y]=canvasXY(e);
  const sx=start[0], sy=start[1];
  const bx=Math.min(sx,x), by=Math.min(sy,y), bw=Math.abs(x-sx), bh=Math.abs(y-sy);
  drag=[drag[0], (bx+bw/2)/img.width/scale, (by+bh/2)/img.height/scale,
        bw/img.width/scale, bh/img.height/scale];
  draw();
});
window.addEventListener('mouseup',e=>{
  if(drag){
    if(drag[3]*img.width<6 || drag[4]*img.height<6) drag=null;   // 误点
    else boxes=[drag];
    drag=null; start=null; draw();
  }
});
document.getElementById('del').onclick=()=>{boxes=[];dirty=true;draw();};
document.getElementById('prev').onclick=()=>{idx=(idx-1+data.images.length)%data.images.length;load();};
document.getElementById('next').onclick=()=>save(true);
document.addEventListener('keydown',e=>{
  if(e.key==='Enter'){save(true);e.preventDefault();}
  else if(e.key==='ArrowRight'){idx=(idx+1)%data.images.length;load();}
  else if(e.key==='ArrowLeft'){idx=(idx-1+data.images.length)%data.images.length;load();}
  else if(e.key==='Delete'||e.key==='Backspace'){boxes=[];dirty=true;draw();e.preventDefault();}
  else if(/^[1-9]$/.test(e.key)){const i=Number(e.key)-1;if(i<data.names.length)clsSel.value=i;draw();}
});
(async()=>{
  data=await (await fetch('/api/list')).json();
  for(let i=0;i<data.names.length;i++){
    const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' '+data.names[i];
    clsSel.appendChild(o);
  }
  clsSel.onchange=()=>{if(boxes.length){boxes[0][0]=Number(clsSel.value);dirty=true;draw();}};
  load();
})();
</script>
</body>
</html>
"""


def list_images(images_dir: Path, labels_dir: Path, out_dir: Path):
    items = []
    for p in sorted(images_dir.glob("*.png")):
        boxes = []
        lp = labels_dir / (p.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    boxes.append([int(parts[0])] + [float(v) for v in parts[1:]])
        items.append({"name": p.name, "boxes": boxes,
                      "done": (out_dir / (p.stem + ".txt")).exists()})
    return items


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
            return
        if self.path == "/api/list":
            items = list_images(Path(ARGS.images), Path(ARGS.labels), Path(ARGS.out))
            body = json.dumps({"names": NAMES, "images": items}, ensure_ascii=False)
            self._send(200, body.encode(), "application/json")
            return
        if self.path.startswith("/img/"):
            name = unquote(self.path[len("/img/"):])
            p = Path(ARGS.images) / name
            if p.exists():
                self._send(200, p.read_bytes(), "image/png")
                return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            name = payload["name"]
            boxes = payload.get("boxes", [])
            out = Path(ARGS.out)
            out.mkdir(parents=True, exist_ok=True)
            lines = []
            for b in boxes:
                c, cx, cy, w, h = b
                lines.append(f"{int(c)} {float(cx):.6f} {float(cy):.6f} "
                             f"{float(w):.6f} {float(h):.6f}")
            (out / (Path(name).stem + ".txt")).write_text("\n".join(lines))
            self._send(200, json.dumps({"ok": True}).encode(), "application/json")
        except Exception as e:                      # noqa: BLE001
            self._send(500, json.dumps({"ok": False, "error": str(e)}).encode(),
                       "application/json")


def main() -> int:
    global ARGS, NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "datasets/p3det_obj/images/val"))
    ap.add_argument("--labels", default=str(ROOT / "datasets/p3det_obj/labels/val"))
    ap.add_argument("--out", default=str(ROOT / "datasets/p3det_obj/labels_manual/val"))
    ap.add_argument("--names", default="airplane,backpack,bicycle,car,cat,chair,cup,"
                                     "desktop,dog,elephant,nailong,penguin,suitcase,tafei,umbrella")
    ap.add_argument("--port", type=int, default=8765)
    ARGS = ap.parse_args()
    NAMES = [s.strip() for s in ARGS.names.split(",") if s.strip()]
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"标注工具：http://127.0.0.1:{ARGS.port}")
    print(f"图片 {ARGS.images}")
    print(f"预标注 {ARGS.labels}")
    print(f"保存到 {ARGS.out}")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
