"""README용 짧은 사용 장면(GIF). tools/screenshots.py와 같이 앱과 헤드리스 크롬을 띄워 둔 채로 돌린다.

폴더 불러오기 → 목소리 고르기 → 찾기 → 해마다 결과를 훑어 내려가는 장면을 찍어 ffmpeg로 GIF를 만든다.

    python tools/demo_gif.py --folder bench/hard --voice 2
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import websockets

from screenshots import CLICK, ROOT, Page

W, H = 1280, 760


class Recorder:
    def __init__(self, page: Page, folder: Path):
        self.p, self.folder, self.n = page, folder, 0

    async def frame(self, hold: int = 1):
        """지금 화면을 hold장으로 저장한다(GIF 한 장 = 0.125초)."""
        r = await self.p.call("Page.captureScreenshot", format="png")
        data = base64.b64decode(r["data"])
        for _ in range(hold):
            (self.folder / f"{self.n:04d}.png").write_bytes(data)
            self.n += 1

    async def scroll_to(self, js_top: str, steps: int = 12):
        """본문 상자를 부드럽게 내린다(Streamlit 본문은 안쪽 상자에서 스크롤된다)."""
        start = await self.p.js("document.querySelector('[data-testid=stMain]').scrollTop")
        end = await self.p.js(f"(() => {{ const m = document.querySelector('[data-testid=stMain]');"
                              f" return m.scrollTop + ({js_top}) - 90; }})()")
        for i in range(1, steps + 1):
            y = start + (end - start) * (1 - (1 - i / steps) ** 2)
            await self.p.js(f"document.querySelector('[data-testid=stMain]').scrollTop = {y}")
            await asyncio.sleep(0.05)
            await self.frame()


async def record(args, out_dir: Path) -> None:
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{args.cdp}/json/list").read())
    page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
    async with websockets.connect(page_ws, max_size=None) as ws:
        p = Page(ws)
        rec = Recorder(p, out_dir)
        await p.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        await p.call("Page.navigate", url=args.url)
        await p.wait_text("불러오기")
        await asyncio.sleep(2)
        await rec.frame(12)
        folder = str((ROOT / args.folder).resolve())
        await p.js(f"""(() => {{
            const inp = [...document.querySelectorAll('input[type=text]')].find(i => i.value.includes('/'));
            Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(inp, {json.dumps(folder)});
            inp.dispatchEvent(new Event('input', {{bubbles: true}}));
            inp.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}}));
        }})()""")
        await asyncio.sleep(1)
        await rec.frame(6)
        await p.js(CLICK.format(label=json.dumps("불러오기")))
        await p.wait_text(f"목소리 {args.voice} 선택")
        await asyncio.sleep(2)
        step2 = "[...document.querySelectorAll('.vm-step')].find(s => s.innerText.startsWith('2')).getBoundingClientRect().top"
        await rec.scroll_to(step2)
        await rec.frame(10)
        await p.js(f"[...document.querySelectorAll('input[type=checkbox]')].find(i => i.closest('label') && "
                   f"i.closest('label').innerText.includes('목소리 {args.voice} 선택')).focus()")
        for kind in ("keyDown", "keyUp"):
            await p.call("Input.dispatchKeyEvent", type=kind, key=" ", code="Space", windowsVirtualKeyCode=32,
                         text=" " if kind == "keyDown" else "")
        await asyncio.sleep(1.5)
        await rec.frame(8)
        await p.js(CLICK.format(label=json.dumps("고른 목소리로 찾기")))
        await p.wait_text("개 해에서 모두")
        await asyncio.sleep(3)
        step3 = "[...document.querySelectorAll('.vm-step')].find(s => s.innerText.startsWith('3')).getBoundingClientRect().top"
        await rec.scroll_to(step3, steps=14)
        await rec.frame(16)
        summary = "[...document.querySelectorAll('p')].find(e => e.innerText.includes('개 해에서 모두')).getBoundingClientRect().top"
        await rec.scroll_to(summary, steps=14)
        await rec.frame(12)
        for _ in range(3):  # 해마다 결과를 훑어 내려간다
            await rec.scroll_to("520", steps=10)
            await rec.frame(6)
        step4 = "[...document.querySelectorAll('.vm-step')].find(s => s.innerText.startsWith('4')).getBoundingClientRect().top"
        await rec.scroll_to(step4, steps=14)
        await rec.frame(20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8517")
    ap.add_argument("--cdp", type=int, default=9333)
    ap.add_argument("--folder", default="bench/hard")
    ap.add_argument("--voice", type=int, default=2)
    ap.add_argument("--width", type=int, default=880, help="GIF 가로 픽셀")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo.gif"))
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="vm-gif-"))
    try:
        asyncio.run(record(args, tmp))
        pal = tmp / "palette.png"
        scale = f"fps=8,scale={args.width}:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "8", "-i", str(tmp / "%04d.png"),
                        "-vf", f"{scale},palettegen=max_colors=96", str(pal)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "8", "-i", str(tmp / "%04d.png"), "-i", str(pal),
                        "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", str(args.out)], check=True)
        print("저장", args.out, f"{Path(args.out).stat().st_size / 1e6:.1f}MB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
