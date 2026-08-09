#!/bin/bash
# Provision the always-on capture VM (e2-small, ~$13/mo) that records the
# configured cameras on a cron schedule and uploads clips to the GCS bucket.
#
# Usage:  ./scripts/gcp/provision_capture_vm.sh [capture-seconds] [cameras]
#   e.g.  ./scripts/gcp/provision_capture_vm.sh 60 tva43
#
# Reads GCP_PROJECT_ID, GCP_ZONE, GCS_BUCKET, CCTV_INVENTORY_URL from .env.
# Re-running with new values updates metadata; reboot the VM to apply.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

CAPTURE_SECONDS="${1:-60}"
CAMERAS="${2:-tva43}"
VM_NAME="tfa-capture"

if gcloud compute instances describe "$VM_NAME" --zone "$GCP_ZONE" \
     --project "$GCP_PROJECT_ID" >/dev/null 2>&1; then
  echo "updating metadata on existing $VM_NAME (reboot to apply)"
  gcloud compute instances add-metadata "$VM_NAME" --zone "$GCP_ZONE" \
    --project "$GCP_PROJECT_ID" \
    --metadata "cctv-inventory-url=${CCTV_INVENTORY_URL},gcs-bucket=${GCS_BUCKET},capture-seconds=${CAPTURE_SECONDS},capture-cameras=${CAMERAS}"
  exit 0
fi

gcloud compute instances create "$VM_NAME" \
  --project "$GCP_PROJECT_ID" \
  --zone "$GCP_ZONE" \
  --machine-type e2-small \
  --image-family debian-12 \
  --image-project debian-cloud \
  --boot-disk-size 20GB \
  --scopes storage-rw \
  --metadata-from-file startup-script=scripts/gcp/capture_vm_startup.sh \
  --metadata "cctv-inventory-url=${CCTV_INVENTORY_URL},gcs-bucket=${GCS_BUCKET},capture-seconds=${CAPTURE_SECONDS},capture-cameras=${CAMERAS}"

echo "created $VM_NAME. Logs: gcloud compute ssh $VM_NAME --zone $GCP_ZONE -- tail -f /var/log/tfa-capture.log"
