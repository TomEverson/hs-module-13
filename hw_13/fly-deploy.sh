#!/usr/bin/env bash
# Deploy hw-13 as ONE Fly.io app.
# Run from the hw_13/ directory: bash fly-deploy.sh
set -e

APP="tomeverson17-gmail-com-harbour-compose"
REGION="ams"

echo "Creating persistent volume (skipped if already exists)..."
fly volumes list --app $APP 2>/dev/null | grep -q app_data \
  || fly volumes create app_data --app $APP --size 3 --region $REGION --yes

echo "Deploying..."
fly deploy --config fly.toml

echo ""
echo "Done! Public URL: https://${APP}.fly.dev"
echo ""
echo "Test:"
echo "  curl -X POST https://${APP}.fly.dev/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\": \"buy cheap watches now\"}'"
echo ""
echo "  curl -X POST https://${APP}.fly.dev/get_message \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"What is machine learning?\"}'"
