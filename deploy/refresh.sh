#!/bin/sh
# 매주 한 번 기반 이미지와 ffmpeg·시스템 패키지의 보안 수정을 받아 다시 빌드한다.
# 코드는 바꾸지 않는다(git pull 하지 않음). 이미지가 바뀌었을 때만 컨테이너를 새로 띄운다(그때 세션은 지워진다).
# cron: 30 5 * * 1 /home/docker/voice-memoir/deploy/refresh.sh >> /home/docker/voice-memoir-refresh.log 2>&1
set -e
cd "$(dirname "$0")/.."
echo "== $(date '+%F %T')"
docker compose -f deploy/compose.yml build --pull --build-arg REFRESH="$(date +%G-%V)" -q
docker compose -f deploy/compose.yml up -d
docker exec deploy-web-1 ffmpeg -version | head -1
