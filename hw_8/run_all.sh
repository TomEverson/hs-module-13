#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
WHITE='\033[1;37m'
NC='\033[0m'

remote_host="158.160.135.246"
private_key="portforward_key"
port_file="/tmp/random_port_hw8.txt"

# Copy portforward_key from hw_6 if not present
if [[ ! -f "$private_key" ]]; then
  cp ../hw_6/portforward_key "$private_key"
  chmod 600 "$private_key"
fi

echo "Loading or generating random port..."
if [[ -f "$port_file" ]]; then
  random_port=$(cat "$port_file")
  echo "Loaded port: $random_port"
else
  random_port=$(awk -v min=1024 -v max=65535 'BEGIN{srand(); print int(min+rand()*(max-min+1))}')
  echo "$random_port" > "$port_file"
  echo "Generated port: $random_port"
fi

echo "Installing dependencies..."
uv sync

echo "Starting SSH tunnel..."
chmod 600 "$private_key"
ssh -f -i "$private_key" -N -R "0.0.0.0:$random_port:localhost:8000" "forwarduser@$remote_host"
if [[ $? -eq 0 ]]; then
  echo -e "${GREEN}SSH tunnel started.${NC}"
else
  echo -e "${RED}Failed to start SSH tunnel.${NC}"
  exit 1
fi

echo -e "${WHITE}Starting FastAPI server on port 8000...${NC}"
uv run python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

echo ""
echo "============================================"
echo -e "Your service URL for registration:"
echo -e "  ${GREEN}http://$remote_host:$random_port${NC}"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop."
trap "kill $SERVER_PID 2>/dev/null; exit 0" INT
wait $SERVER_PID
