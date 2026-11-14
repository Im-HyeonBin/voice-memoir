"""시험·학습에 쓰는 공개 데이터를 원래 출처에서 받는다. 데이터는 저장소에 넣지 않는다.

    python tools/fetch_data.py demo    # 데모·시험 세트용 한국어 테스트 화자 등(약 120MB)
    python tools/fetch_data.py train   # 한국어 추가 학습용 Zeroth 학습 세트·방 울림/소음(약 4.1GB, GPU 서버에서)
    python tools/fetch_data.py test    # 학습에 안 쓰는 시험용 소음(DEMAND)·실제 강연(Pansori)(약 0.7GB)
    python tools/fetch_data.py tse     # 대상 화자 추출의 출발점(ESPnet TD-SpeakerBeam 공개 가중치, 약 65MB)
    python tools/fetch_data.py ami     # 실제 녹음 검증용 AMI 회의 12개와 정답(약 0.7GB)

라이선스는 README '데이터 출처'에 있다. Pansori는 비상업·변경 금지라 시험에만 쓴다.
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets"
AMI_GROUPS = ("ES2004", "IS1009", "TS3003")
GROUPS = {
    "demo": [
        (f"{HF}/Bingsu/zeroth-korean/resolve/main/data/test-00000-of-00001-a41b955a631e582e.parquet",
         "demo_sources/others/zeroth_ko_test.parquet"),
        (f"{HF}/kresnik/zeroth_korean/resolve/main/data/test-00000-of-00001.parquet",
         "demo_sources/others/zeroth_ko_test_with_speakers.parquet"),
    ],
    "train": [
        *[(f"{HF}/kresnik/zeroth_korean/resolve/main/data/train-0000{i}-of-00006.parquet",
           f"data/zeroth/train-0000{i}-of-00006.parquet") for i in range(6)],
        # OpenSLR 28(RIRS_NOISES)과 같은 파일. 원래 주소(openslr.org/resources/28)가 느려 Hugging Face 사본을 쓴다
        (f"{HF}/EaseZh/rirs_noises/resolve/main/rirs_noises.zip", "data/rirs/rirs_noises.zip"),
    ],
    "test": [
        *[(f"https://zenodo.org/records/1227121/files/{e}_16k.zip?download=1", f"data/demand/{e}_16k.zip")
          for e in ("PCAFETER", "PRESTO", "DLIVING", "SPSQUARE", "TBUS")],
        ("https://openslr.elda.org/resources/58/pansori-tedxkr-corpus-1.0.tar.gz",
         "data/pansori/pansori-tedxkr-corpus-1.0.tar.gz"),
    ],
    # 실제 녹음 검증: AMI 회의 말뭉치(CC BY 4.0) 시험 분할의 세 모임. 같은 네 사람이 다른 날 네 번 회의한 것을
    # 방 가운데 마이크 하나(Array1-01)로 녹음했다. 누가 언제 말했는지는 pyannote AMI-diarization-setup의 RTTM(단어 정렬 기준).
    "ami": [
        *[(f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Array1-01.wav",
           f"data/ami/{m[:-1]}/{m}.wav") for g in AMI_GROUPS for m in (g + s for s in "abcd")],
        *[("https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/"
           f"{m}.rttm", f"data/ami/{m[:-1]}/{m}.rttm") for g in AMI_GROUPS for m in (g + s for s in "abcd")],
    ],
    "tse": [
        ("https://huggingface.co/espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw/resolve/main/"
         "exp/enh_train_raw/99epoch.pth", "models/tse/td_speakerbeam_librimix.pth"),
    ],
}


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"있음  {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r받는 중 {dst.name} {done / total:5.1%}", end="", flush=True)
    tmp.replace(dst)  # 다 받은 뒤에만 제 이름으로 바꾼다(중간에 끊겨도 반쪽 파일이 남지 않게)
    print(f"\r받음  {dst.relative_to(ROOT)} ({done / 1e6:.0f}MB)")


def main() -> None:
    groups = sys.argv[1:] or ["demo"]
    for g in groups:
        if g not in GROUPS:
            sys.exit(f"알 수 없는 묶음: {g} (가능: {', '.join(GROUPS)})")
        for url, rel in GROUPS[g]:
            fetch(url, ROOT / rel)
    pansori = ROOT / "data" / "pansori" / "pansori-tedxkr-corpus-1.0.tar.gz"
    if "test" in groups and not (pansori.parent / "pansori-tedxkr-corpus-1.0").exists():
        with tarfile.open(pansori) as t:
            t.extractall(pansori.parent, filter="data")
    if "train" in groups:
        print("다음: python tools/prep_zeroth.py  (학습용 파일로 풀기)")


if __name__ == "__main__":
    main()
