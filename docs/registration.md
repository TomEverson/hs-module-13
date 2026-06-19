# Registration Status

## Attempt 1 (2026-06-19)

- **Name:** Tom
- **Bot/Classifier:** echobot
- **Type:** Classifier
- **Endpoint:** https://lloyd-highway-teenage-headphones.trycloudflare.com

### Test Result: ✅
```
POST /predict → 200 OK
Response: {"id":"...","is_bot_probability":0.88}
```

### Registration Result: ❌
```
Error: "Error while registering classifier. Please check the data"
Backend URL: http://backend:8321/register_classifier
```

### Possible Causes
1. Email `tom@example.com` may be rejected as invalid
2. The `echobot` name may already be taken
3. The classifier returns random probability (not real ML predictions)
4. Server-side validation requiring specific data format

## Next Steps
- Try registering as a **Bot** instead (echobot echoes messages, which is bot behavior)
- Use a real email address
- Try a unique classifier name
