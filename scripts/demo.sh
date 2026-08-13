#!/bin/bash
# One-command reviewer demo: local Kafka -> replayed vehicle events -> windowed
# volume table + alerts, with no cloud, no GPU and no camera access needed.
# Brings up the broker, recreates both topics, replays the recorded 15-minute
# capture and runs the stream processor against it, then prints the run against
# the numbers this capture is known to produce.
#
# Usage:  ./scripts/demo.sh [--speed N] [--no-fault] [--window-seconds N]
#   e.g.  ./scripts/demo.sh --speed 0        # replay as fast as possible
#
# The replay injects a second, synthetic camera (tva43_mirror) that goes silent
# five minutes in. It is a copy of the real capture under a different id, and is
# the only way a single-camera recording can exercise the camera_stale rule.
set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE_FILE=docker/compose.local-kafka.yml
KAFKA=tfa-kafka
REGISTRY=tfa-schema-registry
EVENTS_TOPIC=${TOPIC_VEHICLE_EVENTS:-vehicle.events}
ALERTS_TOPIC=${TOPIC_TRAFFIC_ALERTS:-traffic.alerts}
PARTITIONS=3
VOLUME_CSV=outputs/volume_demo_60s.csv
ALERTS_JSONL=outputs/alerts.jsonl
MIRROR_CAMERA=tva43_mirror
DROP_AFTER=300

SPEED=60
WINDOW_SECONDS=60
FAULT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --speed) SPEED=$2; shift 2 ;;
    --window-seconds) WINDOW_SECONDS=$2; shift 2 ;;
    --no-fault) FAULT=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# uv is the documented path, but the perception extras do not install on every
# machine; a prepared .venv wins so the demo runs on the laptops we have.
if [ -n "${PYTHON:-}" ]; then
  PY=($PYTHON)
elif [ -x .venv/bin/python ]; then
  PY=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python3)
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
kt() { docker exec "$KAFKA" kafka-topics --bootstrap-server localhost:9092 "$@"; }

# Fail early and legibly: `uv run` resolves the whole project, whose perception
# extras (torch, ultralytics) have no wheel on every platform, and the resulting
# error says nothing about this demo. The streaming path needs three packages.
if ! "${PY[@]}" -c "import confluent_kafka, pydantic, dotenv" >/dev/null 2>&1; then
  cat >&2 <<EOF
Cannot run: ${PY[*]} is missing the streaming dependencies.
This demo needs three packages, not the perception stack:

  uv venv --python 3.11
  uv pip install "confluent-kafka[avro,schemaregistry]" pydantic python-dotenv

On an Intel Mac add "cryptography==46.0.3" to that install. Note the project
pins Python 3.11; a system python3 of 3.13 has no cryptography wheel here.

Then re-run ./scripts/demo.sh, or point it at an interpreter with
PYTHON=/path/to/python ./scripts/demo.sh
EOF
  exit 1
fi

step "1/5  Starting local Kafka"
docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop first." >&2; exit 1; }
cold=0
if [ -z "$(docker ps -aq -f name="^${KAFKA}$")" ]; then
  docker compose -f "$COMPOSE_FILE" up -d
  cold=1
else
  # These containers are restart=no, so a reboot leaves them stopped but intact.
  docker start "$KAFKA" "$REGISTRY" >/dev/null
fi
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$KAFKA" 2>/dev/null)" = healthy ]; do
  sleep 2
done
# The registry depends on the broker and exits if it wins the race; it has no
# healthcheck of its own, so poll the API instead.
docker start "$REGISTRY" >/dev/null 2>&1 || true
until curl -sf http://localhost:8081/subjects >/dev/null 2>&1; do sleep 2; done
echo "broker healthy, schema registry responding"

step "2/5  Recreating topics with $PARTITIONS partitions"
# Never auto-created, and never reused: a topic left over from an earlier run
# would replay stale events and make the counts below unreproducible.
for topic in "$EVENTS_TOPIC" "$ALERTS_TOPIC"; do
  kt --delete --topic "$topic" >/dev/null 2>&1 || true
done
sleep 5
for topic in "$EVENTS_TOPIC" "$ALERTS_TOPIC"; do
  kt --create --topic "$topic" --partitions "$PARTITIONS" --replication-factor 1 2>/dev/null
done
kt --list | grep -E "^($EVENTS_TOPIC|$ALERTS_TOPIC)$"

step "3/5  Replaying the recorded capture"
if [ "$cold" = 1 ]; then
  # On a brand-new broker the idempotent producer's first PID request is what
  # triggers the transaction coordinator to load, so it logs one "Coordinator
  # load in progress: retrying" warning and then succeeds. Expected, not a fault:
  # the delivery count below is the thing to read.
  echo "(a first-run 'Coordinator load in progress' warning here is expected)"
fi
replay_args=(--speed "$SPEED")
if [ "$FAULT" = 1 ]; then
  replay_args+=(--mirror-camera "$MIRROR_CAMERA" --drop-after "$DROP_AFTER")
fi
"${PY[@]}" -m src.replay.producer "${replay_args[@]}" &
replay_pid=$!

step "4/5  Processing the stream into ${WINDOW_SECONDS}s event-time windows"
# Windows close on event time only, so the table is identical whether the replay
# is paced at 60x or dumped at --speed 0.
"${PY[@]}" -m src.streaming.consumer \
  --from-beginning --reset-outputs \
  --window-seconds "$WINDOW_SECONDS" \
  --idle-timeout 15 \
  --volume-csv "$VOLUME_CSV" \
  --alerts-jsonl "$ALERTS_JSONL"
wait "$replay_pid"

step "5/5  Result"
rows=$(($(wc -l < "$VOLUME_CSV") - 1))
# The alert log is only created when something fires, so a clean run has no file.
if [ -f "$ALERTS_JSONL" ]; then
  alerts=$(wc -l < "$ALERTS_JSONL" | tr -d ' ')
else
  alerts=0
fi
echo "volume table : $VOLUME_CSV ($rows rows)"
echo "alert log    : $ALERTS_JSONL ($alerts alerts)"
# Counts are deterministic, but two cameras close their windows on independent
# watermarks, so the rows interleave differently as pacing shifts arrival order.
# Checksum the sorted table: that is the part a rerun must reproduce exactly.
echo "checksum     : $(sort "$VOLUME_CSV" | shasum | cut -d' ' -f1) (order-independent)"
echo
echo "counted per camera:"
"${PY[@]}" - "$VOLUME_CSV" <<'EOF'
import csv, collections, sys
totals = collections.Counter()
for row in csv.DictReader(open(sys.argv[1])):
    totals[row["camera_id"]] += int(row["count"])
for camera, total in sorted(totals.items()):
    print(f"  {camera:<14} {total}")
EOF
if [ "$alerts" -gt 0 ]; then
  echo
  echo "alerts raised (also published to $ALERTS_TOPIC):"
  "${PY[@]}" - "$ALERTS_JSONL" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    alert = json.loads(line)
    print(f"  [{alert['alert_type']}] {alert['message']}")
EOF
fi
echo
if [ "$FAULT" = 1 ] && [ "$SPEED" != 0 ]; then
  cat <<'EOF'
expected for this capture:
  tva43          2401   (real, 32 windows)
  tva43_mirror    794   (synthetic, silenced 300s in, 12 windows)
  1 alert: camera_stale on tva43_mirror
EOF
fi
echo "Partition spread (key=camera_id keeps a camera's events ordered on one partition):"
docker exec "$KAFKA" kafka-get-offsets --bootstrap-server localhost:9092 --topic "$EVENTS_TOPIC" 2>/dev/null | sed 's/^/  /'
