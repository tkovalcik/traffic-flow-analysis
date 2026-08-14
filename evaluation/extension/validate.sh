#!/bin/bash
# Controlled-comparison harness: replay one recorded input under varied
# conditions and assert the pipeline produces the same result every time.
#
# The comparison holds the input fixed (data/sample/replay_tva43_15min.jsonl)
# and varies only how the events arrive: dumped as fast as the broker accepts
# them, or paced at 60x event time. The named metric is the SHA-1 of the
# sorted volume table. Windows close on event time, so pacing must not move it.
#
# A third run drops the injected fault to show the fault is what produces the
# alert, and that the real camera's counts are untouched by its presence.
#
# Usage:  ./evaluation/extension/validate.sh [--include-cold]
#   --include-cold  destroy and rebuild the broker first (adds ~75s), proving
#                   the result does not depend on accumulated broker state
set -euo pipefail

cd "$(dirname "$0")/../.."

COMPOSE_FILE=docker/compose.local-kafka.yml
VOLUME_CSV=outputs/volume_demo_60s.csv
ALERTS_JSONL=outputs/alerts.jsonl

# Committed baselines. These are the values the pipeline produced when the
# reviewer demo was first verified; every run below must reproduce them.
EXPECT_CHECKSUM_FAULT=1bd6cba1c010d82951c9c340428dab30dbcc40ff
EXPECT_CHECKSUM_NOFAULT=c08531859140dc98ae60a660e96b93bc241fe31d
EXPECT_ROWS_FAULT=62
EXPECT_ROWS_NOFAULT=46
EXPECT_TVA43=2401
EXPECT_MIRROR=794
EXPECT_ALERTS_FAULT=1
EXPECT_ALERTS_NOFAULT=0
EXPECT_ALERT_IDENTITY=camera_stale:tva43_mirror

INCLUDE_COLD=0
[ "${1:-}" = "--include-cold" ] && INCLUDE_COLD=1

# Keep the two broker conditions side by side rather than overwriting: "same
# result on a warm broker" and "same result on one built from nothing" are
# different claims and the saved evidence should show both.
if [ "$INCLUDE_COLD" = 1 ]; then
  RESULTS=evaluation/extension/results/cold-broker
else
  RESULTS=evaluation/extension/results/warm-broker
fi

mkdir -p "$RESULTS"
pass=0
fail=0

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
check() {
  # check <label> <actual> <expected>
  if [ "$2" = "$3" ]; then
    printf '  \033[32mPASS\033[0m  %-38s %s\n' "$1" "$2"
    pass=$((pass + 1))
  else
    printf '  \033[31mFAIL\033[0m  %-38s got %s, expected %s\n' "$1" "$2" "$3"
    fail=$((fail + 1))
  fi
}

# Pull a labelled figure back out of a demo transcript. demo.sh prints these
# for a human to read; parsing them here keeps one source of truth for the
# numbers rather than reimplementing the counting.
checksum_of() { grep '^checksum' "$1" | awk '{print $3}'; }
rows_of()     { grep '^volume table' "$1" | sed 's/.*(\([0-9]*\) rows).*/\1/'; }
alerts_of()   { grep '^alert log' "$1" | sed 's/.*(\([0-9]*\) alerts).*/\1/'; }
camera_of()   { awk -v cam="$2" '$1 == cam {print $2}' "$1" | head -1; }

# The alert's identity is stable; its reported duration is not. The gap is
# stream_time (max event time seen across all cameras) minus this camera's last
# seen event, both as of whatever the consumer has read so far — so partition
# read skew moves it. Measured 359s in three of four conditions and 840s on a
# cold broker at --speed 0. Assert the parts that are claims, not that one.
alert_identity_of() {
  [ -f "$1" ] || { echo "(no alert file)"; return; }
  # stdlib json only, so the system interpreter is enough here.
  python3 -c "
import json, sys
alert = json.load(open(sys.argv[1]))
print(f\"{alert['alert_type']}:{alert['camera_id']}\")
" "$1"
}

run_demo() {
  # run_demo <tag> <demo.sh args...>
  local tag=$1; shift
  step "Run: $tag"
  # Keep the colour on screen but strip it out of the saved transcript: these
  # are graded artifacts and escape codes render as noise in a text editor.
  local raw="$RESULTS/.transcript-$tag.raw"
  ./scripts/demo.sh "$@" 2>&1 | tee "$raw"
  sed $'s/\033\\[[0-9;]*m//g' "$raw" > "$RESULTS/transcript-$tag.txt"
  rm -f "$raw"
  cp "$VOLUME_CSV" "$RESULTS/volume-$tag.csv"
  # The alert log only exists when something fired; absence is a real result.
  if [ -f "$ALERTS_JSONL" ]; then
    cp "$ALERTS_JSONL" "$RESULTS/alerts-$tag.jsonl"
  else
    rm -f "$RESULTS/alerts-$tag.jsonl"
  fi
}

if [ "$INCLUDE_COLD" = 1 ]; then
  step "Destroying the broker (cold-start condition)"
  docker compose -f "$COMPOSE_FILE" down -v
fi

# Condition A and B differ only in arrival pacing. Same file, same flags.
run_demo speed0  --speed 0
run_demo speed60 --speed 60
# Control: same file, fault injection removed.
run_demo nofault --speed 0 --no-fault

step "Assertions"

a_sum=$(checksum_of "$RESULTS/transcript-speed0.txt")
b_sum=$(checksum_of "$RESULTS/transcript-speed60.txt")
c_sum=$(checksum_of "$RESULTS/transcript-nofault.txt")

echo "Controlled comparison — identical input, varied arrival pacing:"
check "speed 0 checksum" "$a_sum" "$EXPECT_CHECKSUM_FAULT"
check "speed 60 checksum" "$b_sum" "$EXPECT_CHECKSUM_FAULT"
check "speed 0 == speed 60" "$a_sum" "$b_sum"

echo
echo "Counts unchanged across pacing:"
check "speed 0 rows" "$(rows_of "$RESULTS/transcript-speed0.txt")" "$EXPECT_ROWS_FAULT"
check "speed 60 rows" "$(rows_of "$RESULTS/transcript-speed60.txt")" "$EXPECT_ROWS_FAULT"
check "speed 0 tva43 count" "$(camera_of "$RESULTS/transcript-speed0.txt" tva43)" "$EXPECT_TVA43"
check "speed 60 tva43 count" "$(camera_of "$RESULTS/transcript-speed60.txt" tva43)" "$EXPECT_TVA43"
check "speed 0 mirror count" "$(camera_of "$RESULTS/transcript-speed0.txt" tva43_mirror)" "$EXPECT_MIRROR"
check "speed 60 mirror count" "$(camera_of "$RESULTS/transcript-speed60.txt" tva43_mirror)" "$EXPECT_MIRROR"

echo
echo "Fault detection — the injected silence is what raises the alert:"
check "alerts with fault" "$(alerts_of "$RESULTS/transcript-speed0.txt")" "$EXPECT_ALERTS_FAULT"
check "alerts without fault" "$(alerts_of "$RESULTS/transcript-nofault.txt")" "$EXPECT_ALERTS_NOFAULT"
check "speed 0 alert identity" "$(alert_identity_of "$RESULTS/alerts-speed0.jsonl")" "$EXPECT_ALERT_IDENTITY"
check "speed 60 alert identity" "$(alert_identity_of "$RESULTS/alerts-speed60.jsonl")" "$EXPECT_ALERT_IDENTITY"
echo "  note: the alert's reported gap duration is deliberately not asserted"
echo "        (359s here, 840s cold at --speed 0) — see README 'Limitations'."

echo
echo "Control run — removing the fault leaves the real camera untouched:"
check "no-fault checksum" "$c_sum" "$EXPECT_CHECKSUM_NOFAULT"
check "no-fault rows" "$(rows_of "$RESULTS/transcript-nofault.txt")" "$EXPECT_ROWS_NOFAULT"
check "no-fault tva43 count" "$(camera_of "$RESULTS/transcript-nofault.txt" tva43)" "$EXPECT_TVA43"

# The control run is last, so it leaves outputs/ holding the 46-row no-fault
# table and an empty alert log — not the representative artifact the repo README
# documents. Put the fault run back so the two agree.
step "Restoring outputs/ to the representative run"
cp "$RESULTS/volume-speed60.csv" "$VOLUME_CSV"
cp "$RESULTS/alerts-speed60.jsonl" "$ALERTS_JSONL"
echo "$VOLUME_CSV ($(($(wc -l < "$VOLUME_CSV") - 1)) rows), $ALERTS_JSONL ($(wc -l < "$ALERTS_JSONL" | tr -d ' ') alert)"

step "Summary"
{
  echo "run date        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host            : $(uname -srm)"
  echo "cold broker     : $([ "$INCLUDE_COLD" = 1 ] && echo yes || echo 'no (warm)')"
  echo "input           : data/sample/replay_tva43_15min.jsonl"
  echo
  echo "metric          : SHA-1 of the sorted volume table"
  echo "speed 0         : $a_sum"
  echo "speed 60        : $b_sum"
  echo "no-fault        : $c_sum"
  echo
  echo "checks passed   : $pass"
  echo "checks failed   : $fail"
} | tee "$RESULTS/summary.txt"

echo
if [ "$fail" -eq 0 ]; then
  printf '\033[32mAll %d checks passed.\033[0m Artifacts in %s/\n' "$pass" "$RESULTS"
else
  printf '\033[31m%d of %d checks failed.\033[0m\n' "$fail" "$((pass + fail))"
  exit 1
fi
