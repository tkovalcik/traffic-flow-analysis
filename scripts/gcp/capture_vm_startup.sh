#!/bin/bash
# Startup script for the tfa-capture VM: installs the capture stack and a cron
# schedule that records the configured cameras and uploads clips to GCS.
# Configuration arrives via instance metadata (set by provision_capture_vm.sh):
#   cctv-inventory-url, gcs-bucket, capture-seconds, capture-cameras
set -euo pipefail

md() { curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

apt-get update -y
apt-get install -y python3-venv git

if [ ! -d /opt/tfa ]; then
  git clone https://github.com/tkovalcik/traffic-flow-analysis /opt/tfa
else
  git -C /opt/tfa pull --ff-only
fi

if [ ! -d /opt/tfa-venv ]; then
  python3 -m venv /opt/tfa-venv
fi
# Capture-only footprint: no torch/ultralytics on this box.
/opt/tfa-venv/bin/pip install --quiet --upgrade \
  opencv-python-headless numpy requests python-dotenv google-cloud-storage pydantic

cat > /opt/tfa/.env <<EOF
CCTV_INVENTORY_URL=$(md cctv-inventory-url)
GCS_BUCKET=$(md gcs-bucket)
EOF

SECONDS_PER_CLIP=$(md capture-seconds)
CAMERAS=$(md capture-cameras)

# 6x/day at 03,07,11,15,19,23 UTC = 20:00,00:00,04:00,08:00,12:00,16:00 PDT —
# covers night, both commutes, midday, and evening.
cat > /etc/cron.d/tfa-capture <<EOF
0 3,7,11,15,19,23 * * * root for cam in ${CAMERAS//,/ }; do cd /opt/tfa && /opt/tfa-venv/bin/python -m src.replay.record --camera \$cam --seconds ${SECONDS_PER_CLIP} --upload >> /var/log/tfa-capture.log 2>&1; done
EOF
chmod 644 /etc/cron.d/tfa-capture
echo "tfa-capture ready: ${CAMERAS} for ${SECONDS_PER_CLIP}s, 6x/day" >> /var/log/tfa-capture.log
