# Twilio SMS Integration Setup Guide

This guide walks you through setting up SMS forwarding so users can send suspicious messages to your phone number and receive instant spam/smishing analysis.

## Overview

**How it works:**
1. User receives suspicious SMS on their phone
2. User forwards it to your Twilio number (e.g., +1-555-SPAM-123)
3. Twilio sends the message to your API webhook
4. API analyzes the message using the spam detection model
5. API sends back the classification result
6. User receives instant analysis on their phone

**Cost:** ~$0.015 per message check ($0.0075 to receive + $0.0075 to send)

---

## Step 1: Create Twilio Account

1. Go to [https://www.twilio.com/try-twilio](https://www.twilio.com/try-twilio)
2. Sign up for a free account
3. Verify your email and phone number
4. You'll get **free trial credits** (~$15) to test the service

---

## Step 2: Get a Phone Number

1. Log in to [Twilio Console](https://console.twilio.com)
2. Navigate to **Phone Numbers** → **Manage** → **Buy a number**
3. Select your country (e.g., United States)
4. Filter by capabilities: Check **SMS**
5. Choose a number you like (tip: pick something memorable like with "SPAM" pattern)
6. Click **Buy** (costs $1-2/month)

**Your phone number will look like:** `+1-555-123-4567`

---

## Step 3: Get Your Credentials

1. Go to [Twilio Console Dashboard](https://console.twilio.com)
2. Find your **Account SID** and **Auth Token** in the dashboard
3. Click the eye icon to reveal the Auth Token

**Save these securely** - you'll need them in the next step.

---

## Step 4: Configure Your API

### Option A: Environment Variables (Recommended)

Create a `.env` file in the project root:

```bash
# Copy from example
cp .env.example .env
```

Edit `.env` and add your Twilio credentials:

```bash
# Twilio SMS Integration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567  # Your Twilio number (E.164 format)
```

**Important Notes:**
- Use E.164 format for phone numbers: `+[country code][number]`
- No spaces or dashes in the phone number
- Keep the `.env` file secure - never commit it to git (it's already in `.gitignore`)

### Option B: System Environment Variables

```bash
export TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export TWILIO_AUTH_TOKEN=your_auth_token_here
export TWILIO_PHONE_NUMBER=+15551234567
```

---

## Step 5: Deploy Your API

### Local Development (Testing)

For local testing, you'll need to expose your local server to the internet. Use **ngrok**:

```bash
# Start your API
uvicorn src.api.main:app --reload --port 8000

# In another terminal, start ngrok
ngrok http 8000
```

You'll get a public URL like: `https://abc123.ngrok.io`

**Your webhook URL will be:** `https://abc123.ngrok.io/sms/incoming`

### Production Deployment

Deploy your API to a cloud provider:

**Option 1: Docker on any server**
```bash
docker-compose up -d
```

**Option 2: Cloud platforms**
- **AWS**: EC2, ECS, or Lambda
- **Google Cloud**: Cloud Run or Compute Engine
- **Azure**: App Service or Container Instances
- **DigitalOcean**: Droplet or App Platform
- **Heroku**: `git push heroku main`

Make note of your public URL: `https://your-domain.com`

**Your webhook URL will be:** `https://your-domain.com/sms/incoming`

---

## Step 6: Configure Twilio Webhook

1. Go to [Twilio Console](https://console.twilio.com)
2. Navigate to **Phone Numbers** → **Manage** → **Active numbers**
3. Click on your phone number
4. Scroll to **Messaging Configuration**
5. Under **A MESSAGE COMES IN**:
   - Select **Webhook**
   - Enter your webhook URL: `https://your-domain.com/sms/incoming`
   - HTTP Method: **POST**
   - Content Type: **application/x-www-form-urlencoded** (default)
6. Click **Save**

---

## Step 7: Test It!

### Test 1: Forward a Safe Message

1. On your phone, create a test message or find a legitimate 2FA text
2. Forward it to your Twilio number
3. Wait 1-2 seconds
4. You should receive:

```
✅ SAFE (99% confidence)

This message appears legitimate.
```

### Test 2: Forward a Spam Message

Create a fake spam message and forward it:

```
URGENT! Your account has been locked.
Click bit.ly/verify to restore access NOW!
```

You should receive:

```
🚨 SMISHING DETECTED
Confidence: 98%
Risk: HIGH

• URL shortener detected
• Urgent language
• Account verification request

⚠️ DO NOT CLICK LINKS
⚠️ DO NOT REPLY
⚠️ DELETE MESSAGE
```

---

## Troubleshooting

### "Service temporarily unavailable"

**Cause:** API server is down or model not loaded

**Fix:**
1. Check if your API is running: `curl https://your-domain.com/health`
2. Check server logs for errors
3. Verify Twilio webhook URL is correct

### "No response received"

**Cause:** Twilio can't reach your webhook URL

**Fix:**
1. Verify webhook URL is publicly accessible
2. Check URL in browser: should show "Method Not Allowed" (that's OK - GET isn't supported)
3. For local testing, ensure ngrok is running
4. Check firewall settings

### "Authentication failed"

**Cause:** Wrong Twilio credentials

**Fix:**
1. Verify `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` in `.env`
2. Check for spaces or quotes in the values
3. Regenerate Auth Token if needed (in Twilio Console)

### "Invalid phone number format"

**Cause:** Phone number not in E.164 format

**Fix:**
- Must start with `+`
- Include country code: `+1` for US/Canada
- No spaces, dashes, or parentheses
- Example: `+15551234567` ✅
- Not: `(555) 123-4567` ❌

### SMS not forwarding properly

**Cause:** Some phones format forwards differently

**Fix:**
The API automatically strips common forward prefixes like:
- "Fwd:"
- "Forwarded message:"
- "> "

If still having issues, just copy/paste the message content instead of using forward.

---

## Monitoring SMS Traffic

### Check API Stats

```bash
curl https://your-domain.com/stats
```

Response:
```json
{
  "total_requests": 245,
  "spam_detected": 89,
  "ham_detected": 156,
  "spam_rate": 0.36,
  "feedback_received": 12,
  "model_name": "Smart Router",
  "sms_received": 178
}
```

### View Logs

The API logs each SMS received:

```
📱 SMS received from +15551234567
   Message SID: SM1234567890abcdef
   Content: URGENT! Your account has been locked...
   Classification: smishing (0.98)
   Latency: 12.3ms
   Response: 🚨 SMISHING DETECTED...
```

---

## Cost Estimation

### Twilio Pricing (US)

- **Phone number rental**: $1.15/month
- **Incoming SMS**: $0.0075 per message
- **Outgoing SMS**: $0.0075 per message
- **Total per check**: ~$0.015

### Example Monthly Costs

| Usage | Cost |
|-------|------|
| 10 checks/month | $1.30/month |
| 50 checks/month | $2.65/month |
| 100 checks/month | $4.15/month |
| 500 checks/month | $15.65/month |

**For personal use (10-50 checks/month): ~$2-3/month**

---

## Security Best Practices

### 1. Validate Twilio Requests (Optional)

Add request validation to ensure webhooks are actually from Twilio:

```python
from twilio.request_validator import RequestValidator

validator = RequestValidator(auth_token)
if not validator.validate(url, post_vars, signature):
    raise HTTPException(status_code=403, detail="Invalid signature")
```

### 2. Rate Limiting

Add rate limiting to prevent abuse:

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/sms/incoming", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
```

### 3. Store Credentials Securely

- Use environment variables or secrets manager
- Never commit `.env` to git
- Rotate Auth Token regularly
- Use different credentials for dev/prod

### 4. Monitor Usage

Set up alerts for:
- Unexpected spike in SMS volume
- High error rates
- Unusual sender patterns

---

## Advanced Configuration

### Custom Response Format

Edit `src/api/sms_handler.py` to customize response format:

```python
def format_classification_response(...):
    # Customize your response format here
    if is_smishing:
        response = "🚨 DANGER! This is a phishing scam."
    # ...
```

### Multi-Language Support

Add language detection and responses:

```python
from langdetect import detect

language = detect(message)
if language == 'es':
    response = "🚨 SMISHING DETECTADO"
```

### Integrate with Phone Carrier APIs

Some carriers (Verizon, AT&T) have APIs to report spam. You could automatically report detected smishing.

---

## Alternative: Use Email-to-SMS

If you don't want to pay for Twilio, many carriers support email-to-SMS:

### Carrier Gateways

- AT&T: `number@txt.att.net`
- Verizon: `number@vtext.com`
- T-Mobile: `number@tmomail.net`
- Sprint: `number@messaging.sprintpcs.com`

### Setup

1. Configure email server (SendGrid, Mailgun, AWS SES)
2. Add email webhook endpoint to your API
3. Forward suspicious messages to: `check@your-domain.com`
4. API emails back the result

**Pros:** Cheaper for low volume
**Cons:** Slower, less reliable, harder to parse

---

## Next Steps

Now that SMS forwarding is set up:

1. Share your Twilio number with friends/family
2. Add it to your phone contacts as "Spam Checker"
3. Test with various message types
4. Monitor accuracy and collect feedback
5. Consider adding a simple landing page explaining how to use it

---

## Support

**Issues?** Open a GitHub issue or check:
- Twilio Status: [https://status.twilio.com](https://status.twilio.com)
- API Health: `https://your-domain.com/health`
- Logs: `docker logs <container-id>` or server logs

**Documentation:**
- [Twilio SMS Docs](https://www.twilio.com/docs/sms)
- [TwiML Reference](https://www.twilio.com/docs/sms/twiml)
- [Webhook Security](https://www.twilio.com/docs/usage/webhooks/webhooks-security)
