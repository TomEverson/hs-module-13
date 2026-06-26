#!/usr/bin/env bash
# Deploy TomBot to Fly.io as a single app.
# Run from the final/ directory: bash fly-deploy.sh
set -e

APP="tomeverson17-gmail-com-harbour-compose"
REGION="ams"

echo "Creating persistent volume (skipped if already exists)..."
fly volumes list --app $APP 2>/dev/null | grep -q tombot_data \
  || fly volumes create tombot_data --app $APP --size 3 --region $REGION --yes

echo "Deploying..."
fly deploy --config fly.toml

echo ""
echo "Done! Public URL: https://${APP}.fly.dev"
echo ""
echo "Test predict:"
echo "  curl -X POST https://${APP}.fly.dev/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\":\"buy cheap watches now\",\"dialog_id\":\"00000000-0000-0000-0000-000000000001\",\"id\":\"00000000-0000-0000-0000-000000000002\",\"participant_index\":0}'"
echo ""
echo "Test get_message:"
echo "  curl -X POST https://${APP}.fly.dev/get_message \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"dialog_id\":\"00000000-0000-0000-0000-000000000001\",\"last_msg_text\":\"hey what are you up to\"}'"
