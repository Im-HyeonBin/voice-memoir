"""README용 화면 캡처. 앱을 띄워 둔 채로 헤드리스 크롬을 조종해 폴더 불러오기 → 목소리 고르기 → 찾기를 누르고
단계별로 찍는다(Chrome DevTools Protocol, 추가 설치 없이 websockets만 씀).

    streamlit run app.py --server.port 8517 &
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --remote-debugging-port=9333 \
        --user-data-dir=/tmp/vm-shot about:blank &
    python tools/screenshots.py --folder bench/hard --voice 2
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def wait_text(self, text: str, timeout: float = 180):
        for _ in range(int(timeout * 2)):
            if await self.js(f"document.body.innerText.includes({json.dumps(text)})"):
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(text)

    async def shot(self, path: Path, top: float, bottom: float, width: int):
        # 옆 막대(설정)는 빼고 본문 기둥만 찍는다
        left, right = await self.js("(() => { const r = document.querySelector('[data-testid=stMainBlockContainer]')"
                                    ".getBoundingClientRect(); return [r.left, r.right]; })()")
        x0, x1 = max(0, left - 8), min(width, right + 8)
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                            clip={"x": x0, "y": top, "width": x1 - x0, "height": bottom - top, "scale": 1})
        path.write_bytes(base64.b64decode(r["data"]))
        print("저장", path.relative_to(ROOT), f"{bottom - top:.0f}px")


def rect(selector_js: str) -> str:
    return f"(() => {{ const r = ({selector_js}).getBoundingClientRect(); return [r.top, r.bottom]; }})()"


STEP = "[...document.querySelectorAll('.vm-step')].find(s => s.innerText.startsWith('{n}'))"
CLICK = "[...document.querySelectorAll('button')].find(b => b.innerText.trim() === {label}).click()"


async def main(args) -> None:
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{args.cdp}/json/list").read())
    page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
    out = ROOT / "docs" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    async with websockets.connect(page_ws, max_size=None) as ws:
        p = Page(ws)
        W = 1280
        # 앱 본문은 안쪽 상자에서 스크롤되므로, 창을 길게 잡아 한 번에 다 그리게 한다
        await p.call("Emulation.setDeviceMetricsOverride", width=W, height=args.height, deviceScaleFactor=2, mobile=False)
        await p.call("Page.navigate", url=args.url)
        await p.wait_text("불러오기")
        await asyncio.sleep(2)
        await p.shot(out / "0_first_screen.png", 0, 820, W)
        folder = str((ROOT / args.folder).resolve())
        await p.js(f"""(async () => {{
            const inp = [...document.querySelectorAll('input[type=text]')].find(i => i.value.includes('/'));
            Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(inp, {json.dumps(folder)});
            inp.dispatchEvent(new Event('input', {{bubbles: true}}));
            inp.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}}));
            await new Promise(r => setTimeout(r, 800));
            {CLICK.format(label=json.dumps('불러오기'))};
        }})()""")
        await p.wait_text(f"목소리 {args.voice} 선택")
        await asyncio.sleep(3)
        top2 = (await p.js(rect(STEP.format(n=1))))[0] - 24
        bottom2 = (await p.js(rect("[...document.querySelectorAll('button')].find(b => b.innerText.trim() === '고른 목소리로 찾기')")))[1] + 32
        await p.shot(out / "1_folder_and_voices.png", max(0, top2), bottom2, W)

        # Streamlit 체크박스는 마우스 흉내로는 안 눌려서 포커스 후 스페이스 키를 보낸다
        await p.js(f"[...document.querySelectorAll('input[type=checkbox]')].find(i => i.closest('label') && "
                   f"i.closest('label').innerText.includes('목소리 {args.voice} 선택')).focus()")
        for kind in ("keyDown", "keyUp"):
            await p.call("Input.dispatchKeyEvent", type=kind, key=" ", code="Space", windowsVirtualKeyCode=32,
                         text=" " if kind == "keyDown" else "")
        await asyncio.sleep(2)
        await p.js(CLICK.format(label=json.dumps("고른 목소리로 찾기")))
        await p.wait_text("개 해에서 모두")
        await asyncio.sleep(4)
        top3 = (await p.js(rect(STEP.format(n=3))))[0] - 24
        await p.shot(out / "2_results_by_year.png", top3, top3 + args.result_height, W)
        top4, _ = await p.js(rect(STEP.format(n=4)))
        end = await p.js(rect("[...document.querySelectorAll('button')].find(b => b.innerText.includes('모두 받기'))"))
        await p.shot(out / "3_download_list.png", top4 - 24, end[1] + 32, W)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8517")
    ap.add_argument("--cdp", type=int, default=9333)
    ap.add_argument("--folder", default="bench/hard")
    ap.add_argument("--voice", type=int, default=2, help="찾을 사람 목소리 번호(폴더마다 다름)")
    ap.add_argument("--height", type=int, default=7000)
    ap.add_argument("--result-height", type=int, default=1500)
    asyncio.run(main(ap.parse_args()))
