# 목소리 발굴 온라인 서비스. CPU로만 돈다(GPU를 쓰면 집 서버 전체가 멈춘 적이 있다).
# 모델 가중치는 이미지에 넣지 않고 읽기 전용으로 붙인다(deploy/compose.yml).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY visual.py ./
COPY vm/ vm/
COPY web/ web/

RUN useradd --system --uid 10001 --home /tmp vm && mkdir -p /data && chown vm /data
USER vm

ENV VM_DATA=/data VM_DEVICE=cpu VM_THREADS=4 VM_SCAN_WORKERS=1 VM_TRUST_PROXY=1 \
    HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
EXPOSE 8600
HEALTHCHECK --interval=60s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8600/api/health', timeout=4)"
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8600", "--no-access-log", "--no-server-header"]
