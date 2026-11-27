"""온라인 서비스(web/) README용 화면 캡처와 짧은 사용 장면(GIF).

서버와 헤드리스 크롬을 띄워 둔 채로 돌린다. 예시 녹음으로 살펴보기 → 목소리 고르기 → 참고 녹음으로 찾기를 누르며 찍는다.

    .venv/bin/uvicorn web.server:app --port 8600 &
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --remote-debugging-port=9333 \\
        --user-data-dir=/tmp/vm-shot about:blank &
    python tools/web_screenshots.py            # docs/screenshots/web_*.png
    python tools/web_screenshots.py --gif      # docs/web_demo.gif
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

from screenshots import ROOT, Page

OUT = ROOT / "docs" / "screenshots"


async def screen(p: Page, name: str, timeout: float = 240) -> None:
    for _ in range(int(timeout * 2)):
        if await p.js(f"document.body.dataset.screen === {json.dumps(name)}"):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(name)


async def result_ready(p: Page, timeout: float = 300) -> None:
    for _ in range(int(timeout * 2)):
        if await p.js("document.body.dataset.screen === 'result' && document.querySelector('#rSum').textContent.length > 0"
                      " && document.querySelector('#working').hidden"):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError("result")


async def shot(p: Page, path: Path, top: float, bottom: float, width: int) -> None:
    r = await p.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                     clip={"x": 0, "y": top, "width": width, "height": bottom - top, "scale": 1})
    path.write_bytes(base64.b64decode(r["data"]))
    print("저장", path.relative_to(ROOT))


async def start_demo(p: Page, url: str) -> None:
    await p.call("Page.navigate", url=url)
    await asyncio.sleep(1.5)
    await p.js("localStorage.clear(); location.reload()")
    await p.wait_text("예시 녹음으로 해 보기")
    await asyncio.sleep(3.5)  # 첫 화면 파형이 초록으로 바뀌는 것을 기다린다


async def stills(p: Page, url: str) -> None:
    W, H = 1280, 860
    OUT.mkdir(parents=True, exist_ok=True)
    await p.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
    await start_demo(p, url)
    await shot(p, OUT / "web_0_home.png", 0, H, W)

    await p.js("document.querySelector('#demoBtn').click()")
    await screen(p, "pick")
    await asyncio.sleep(1.5)
    await p.js("document.querySelector('#voices .vrow:nth-child(3) .pickbtn').click()")
    await asyncio.sleep(0.5)
    # 화면 아래에 붙어 다니는 '찾기' 막대가 목록을 가리지 않게 캡처할 때만 제자리에 둔다
    bottom = await p.js("(() => { const g = document.querySelector('.go'); g.style.position = 'static';"
                        " return g.getBoundingClientRect().bottom + scrollY + 28; })()")
    await shot(p, OUT / "web_1_pick.png", 0, bottom, W)
    await p.js("document.querySelector('.go').style.position = ''")

    await p.js("document.querySelector('.tabs button[data-tab=ref]').click(); document.querySelector('#demoRefBtn').click()")
    await result_ready(p)
    await asyncio.sleep(1.5)
    await shot(p, OUT / "web_2_result.png", 0, 1500, W)

    await p.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=3, mobile=True)
    await p.js("location.reload()")  # 창을 새로 열어도 결과로 돌아온다
    await result_ready(p)
    await asyncio.sleep(1.5)
    await shot(p, OUT / "web_3_mobile.png", 0, 1500, 390)


async def gif(p: Page, url: str, frames: Path) -> None:
    W, H = 1280, 760
    n = 0

    async def frame(hold: int = 1) -> None:
        nonlocal n
        r = await p.call("Page.captureScreenshot", format="png")
        data = base64.b64decode(r["data"])
        for _ in range(hold):
            (frames / f"{n:04d}.png").write_bytes(data)
            n += 1

    async def scroll(to: float, steps: int = 14) -> None:
        start = await p.js("scrollY")
        for i in range(1, steps + 1):
            await p.js(f"scrollTo(0, {start + (to - start) * (1 - (1 - i / steps) ** 2)})")
            await asyncio.sleep(0.05)
            await frame()

    await p.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    await p.call("Page.navigate", url=url)
    await asyncio.sleep(1.5)
    await p.js("localStorage.clear(); location.reload()")
    await p.wait_text("예시 녹음으로 해 보기")
    for _ in range(16):  # 파형이 초록으로 바뀌는 장면
        await frame()
        await asyncio.sleep(0.2)
    await frame(8)
    await p.js("document.querySelector('#demoBtn').click()")
    while not await p.js("document.body.dataset.screen === 'pick'"):
        await frame()
        await asyncio.sleep(0.8)
    await asyncio.sleep(1)
    await frame(10)
    await p.js("document.querySelector('#voices .vrow:nth-child(3) .pickbtn').click()")
    await frame(8)
    await p.js("document.querySelector('#findBtn').click()")
    while not await p.js("document.body.dataset.screen === 'result' && document.querySelector('#working').hidden"
                         " && document.querySelector('#rSum').textContent.length > 0"):
        await frame()
        await asyncio.sleep(0.8)
    await asyncio.sleep(1)
    await frame(12)
    first_year = await p.js("document.querySelector('#timeline .yr').getBoundingClientRect().top + scrollY - 80")
    await scroll(first_year)
    await frame(12)
    await scroll(first_year + 900, 18)
    await frame(14)


async def main(args) -> None:
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{args.cdp}/json/list").read())
    page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
    async with websockets.connect(page_ws, max_size=None) as ws:
        p = Page(ws)
        if not args.gif:
            await stills(p, args.url)
            return
        tmp = Path(tempfile.mkdtemp(prefix="vm-webgif-"))
        try:
            await gif(p, args.url, tmp)
            pal, scale = tmp / "palette.png", f"fps=8,scale={args.width}:-1:flags=lanczos"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "8", "-i", str(tmp / "%04d.png"),
                            "-vf", f"{scale},palettegen=max_colors=96", str(pal)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "8", "-i", str(tmp / "%04d.png"), "-i", str(pal),
                            "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", args.out], check=True)
            print("저장", args.out, f"{Path(args.out).stat().st_size / 1e6:.1f}MB")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600")
    ap.add_argument("--cdp", type=int, default=9333)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--width", type=int, default=880, help="GIF 가로 픽셀")
    ap.add_argument("--out", default=str(ROOT / "docs" / "web_demo.gif"))
    asyncio.run(main(ap.parse_args()))
