"""Zeroth-Korean 학습 세트(한국인 105명, CC BY 4.0)를 학습하기 좋게 한 파일로 푼다.

parquet 안의 flac을 풀어 16비트 원음 하나(train_audio.int16)로 이어 붙이고, 문장마다
시작 위치·길이·화자 번호를 train_index.npz에 남긴다. 학습할 때는 이 파일을 메모리에 다 올리지 않고
필요한 부분만 읽는다(약 6GB).

    python tools/prep_zeroth.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "zeroth"


def main() -> None:
    out = SRC / "train_audio.int16"
    offsets, lengths, speakers = [], [], []
    pos = 0
    with open(out, "wb") as f:
        for part in sorted(SRC.glob("train-*.parquet")):
            for batch in pq.ParquetFile(part).iter_batches(batch_size=256, columns=["speaker_id", "audio"]):
                for spk, a in zip(batch.column("speaker_id").to_pylist(), batch.column("audio").to_pylist()):
                    x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="int16")
                    if sr != 16000 or x.ndim != 1 or len(x) < 16000 * 2:
                        continue
                    f.write(x.tobytes())
                    offsets.append(pos)
                    lengths.append(len(x))
                    speakers.append(spk)
                    pos += len(x)
            print(f"{part.name}: 누적 문장 {len(offsets)}개, {pos / 16000 / 3600:.1f}시간", flush=True)
    np.savez(SRC / "train_index.npz", offsets=np.array(offsets, np.int64), lengths=np.array(lengths, np.int64),
             speakers=np.array(speakers, np.int64))
    print(f"화자 {len(set(speakers))}명, 문장 {len(offsets)}개")


if __name__ == "__main__":
    main()
