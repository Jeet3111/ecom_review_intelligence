#!/usr/bin/env bash
# Deploy the web app to Google Cloud Run (serverless containers).
# Prerequisites: gcloud CLI installed + `gcloud auth login` + a project with billing / student coupon applied.
# Usage:  PROJECT_ID=my-project ./deploy_gcp.sh
set -euo pipefail
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=my-project ./deploy_gcp.sh}"
REGION="${REGION:-asia-south1}"          # Mumbai - closest region for Indian users
SERVICE="${SERVICE:-review-intelligence}"

[ -f models/spam_model.npz ] || { echo "Train first:  python run_pipeline.py"; exit 1; }

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 512Mi --cpu 1 --max-instances 2 --port 8080

echo
echo "Service URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
