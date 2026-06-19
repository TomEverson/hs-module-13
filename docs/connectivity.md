# Connectivity Guide

## Methods to Expose Local Bot to You Are Bot

### Method 1: Cloudflare Tunnel (Working ✅)

Cloudflare Tunnel is available via `cloudflared` CLI:

```bash
# Start tunnel
cloudflared tunnel --url http://localhost:6872

# Example output URL:
# https://lloyd-highway-teenage-headphones.trycloudflare.com
```

**Pros:** Free, no account needed, works immediately
**Cons:** Temporary URL changes each restart, no uptime guarantee

### Method 2: SSH Reverse Tunnel (from scripts)

The `run_all_linux.sh` / `run_all_windows.ps1` scripts set up a reverse SSH tunnel:

```bash
ssh -f -i portforward_key -N -R 0.0.0.0:{random_port}:localhost:6872 forwarduser@158.160.135.246
```

**Current status (2026-06-19):** The `forwarduser` account on `158.160.135.246` returns "This account is currently not available." The SSH connection to port 22 succeeds but the account has no active shell. Using `-N` flag allows the tunnel to stay open (no remote command needed), but port forwarding to the external interface may be blocked by the server's sshd configuration (`GatewayPorts`).

### Method 3: ngrok

```bash
ngrok http 6872
```

**Available:** ngrok v3.23.2 is installed at `/opt/homebrew/bin/ngrok`
**Note:** Requires ngrok account for persistent URLs

## Required Endpoints for You Are Bot Registration

Your service must expose these endpoints:

### `GET /health`
```json
{"status": "ok"}
```

### `POST /predict`
Request:
```json
{
  "text": "Hello",
  "dialog_id": "00000000-0000-0000-0000-000000000001",
  "id": "00000000-0000-0000-0000-000000000002",
  "participant_index": 0
}
```

Response:
```json
{
  "id": "00000000-0000-0000-0000-000000000003",
  "message_id": "00000000-0000-0000-0000-000000000002",
  "dialog_id": "00000000-0000-0000-0000-000000000001",
  "participant_index": 0,
  "is_bot_probability": 0.42
}
```

### `POST /get_message` (for bots)
Request:
```json
{
  "dialog_id": "00000000-0000-0000-0000-000000000001",
  "last_msg_text": "Hello",
  "last_message_id": "00000000-0000-0000-0000-000000000002"
}
```

Response:
```json
{
  "new_msg_text": "Response text here",
  "dialog_id": "00000000-0000-0000-0000-000000000001"
}
```
