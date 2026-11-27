#!/bin/sh
# 목소리 발굴 컨테이너(브리지 br-voice)에서 밖으로 나가는 새 연결을 막는다.
# 남이 올린 파일을 해석하다 컨테이너가 장악돼도 녹음을 바깥으로 보내거나 집 내부망(Proxmox, 다른 VM)으로
# 넘어가지 못하게 한다. 터널로 들어온 요청의 응답(이미 맺은 연결)은 그대로 둔다.
# 설치: sudo install -m 755 deploy/egress.sh /usr/local/sbin/voice-egress.sh
#       sudo install -m 644 deploy/voice-egress.service /etc/systemd/system/ && sudo systemctl enable --now voice-egress
set -e
add() {  # 같은 규칙이 없을 때만 맨 앞에 넣는다
  chain=$1; shift
  iptables -C "$chain" "$@" 2>/dev/null || iptables -I "$chain" "$@"
}
add DOCKER-USER -i br-voice -m conntrack --ctstate NEW -j DROP   # 컨테이너 → 인터넷·내부망
add INPUT -i br-voice -m conntrack --ctstate NEW -j DROP         # 컨테이너 → 이 서버 자신(SSH 등)
