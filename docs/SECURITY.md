# Security and Credential Protection

## Overview

This document outlines the security measures implemented to protect sensitive credentials and user data in the SMS Spam Detection system.

## Credential Protection

### Twilio API Credentials

**Storage**: All Twilio credentials are stored in `.env` file which is:
- ✅ Listed in `.gitignore` and never committed to version control
- ✅ Loaded via `python-dotenv` package
- ✅ Accessed only through environment variables
- ✅ Never logged or displayed in output

**Required Credentials**:
```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567
```

### Git Protection

**.gitignore Configuration**:
```
# Environment variables
.env
.env.local
```

**Verification**: Run to confirm `.env` was never committed:
```bash
git log --all --full-history -- .env
# Should return empty (no results)
```

### Example Configuration

A safe template is provided in `.env.example` with placeholder values:
```bash
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890
```

## Data Privacy

### Message Storage

**What We Store**:
- Message classification (spam/ham/smishing)
- Confidence score
- Timestamp
- Message length (not content)
- Hash of phone number (not actual number)

**What We DON'T Store**:
- Full message text (only used in-memory for classification)
- Actual phone numbers (only hashed for analytics)
- User identifiable information

**Storage Location**:
- `data/message_history.jsonl` - Classification records only
- `data/daily_stats.json` - Aggregated statistics
- `data/spending_state.json` - Spending tracking

### Data Files in `.gitignore`

```bash
# Data
data/raw/*
data/processed/*
data/feedback/*
data/spending_state.json
data/message_history.jsonl
data/daily_stats.json
```

## API Security

### Rate Limiting

**Spending Monitor** acts as a rate limiter:
- Daily spending limit: $5 (configurable)
- ~333 messages per day maximum
- Automatic service disable when limit exceeded
- SMS alert sent to owner

### Request Validation

**Twilio Webhook Security**:
- Accepts only POST requests with proper form data
- Validates required fields (From, Body, MessageSid)
- Returns XML responses (TwiML format)

**Optional Enhancement** (not yet implemented):
```python
from twilio.request_validator import RequestValidator

# Validate request signature
validator = RequestValidator(auth_token)
if not validator.validate(url, params, signature):
    raise HTTPException(403, "Invalid signature")
```

## Best Practices

### For Developers

1. **Never commit `.env`**
   ```bash
   # If accidentally added:
   git rm --cached .env
   git commit -m "Remove .env from tracking"
   ```

2. **Rotate credentials regularly**
   - Update `TWILIO_AUTH_TOKEN` in Twilio Console
   - Update `.env` file
   - Restart API service

3. **Use environment-specific credentials**
   ```bash
   # Development
   .env.development

   # Production
   .env.production
   ```

4. **Audit access logs**
   ```bash
   # Check API logs for suspicious activity
   tail -f logs/api.log
   ```

### For Deployment

**Environment Variables** (preferred over `.env` in production):
```bash
# Set in production environment
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
export TWILIO_PHONE_NUMBER=...
```

**Secrets Management**:
- **AWS**: Use AWS Secrets Manager or Parameter Store
- **Azure**: Use Azure Key Vault
- **GCP**: Use Google Secret Manager
- **Heroku**: Use Config Vars
- **Docker**: Use Docker secrets

## Monitoring

### Security Alerts

**Spending Monitor** provides protection against:
- Unexpected API usage spikes
- Potential abuse or DoS attempts
- Accidental runaway processes

**Alert Threshold**: $5/day (adjustable via `TWILIO_DAILY_LIMIT`)

### Audit Trail

**Message History** (`data/message_history.jsonl`):
```json
{
  "timestamp": "2026-01-06T10:30:00",
  "classification": "spam",
  "is_smishing": true,
  "confidence": 0.98,
  "via_sms": true,
  "message_length": 145,
  "from_number_hash": 123456789
}
```

**No PII stored** - only hashed phone numbers and metadata.

## Compliance

### GDPR Considerations

- ✅ Minimal data collection
- ✅ No storage of message content
- ✅ Phone numbers hashed (not stored)
- ✅ User can request data deletion

### Data Retention

**Current Policy**:
- Message history: Indefinite (metadata only)
- Daily stats: Archived indefinitely
- No automatic deletion

**To Delete Data**:
```bash
# Remove all tracking data
rm data/message_history.jsonl
rm data/daily_stats.json
rm data/stats_archive.jsonl
```

## Incident Response

### If Credentials Are Compromised

1. **Immediate Actions**:
   ```bash
   # Revoke compromised token in Twilio Console
   # https://console.twilio.com -> Settings -> API Credentials
   ```

2. **Rotate Credentials**:
   - Generate new Auth Token
   - Update `.env` file
   - Restart API service

3. **Check for Abuse**:
   ```bash
   # Review Twilio usage logs
   # https://console.twilio.com/us1/monitor/logs/sms

   # Check local spending state
   cat data/spending_state.json
   ```

4. **Notify Users** (if applicable):
   - Send alert via SMS if service was abused
   - Update stakeholders

### If Repository Is Exposed

1. **Check Git History**:
   ```bash
   git log --all --full-history -- .env
   ```

2. **If `.env` Was Committed**:
   ```bash
   # Remove from history (WARNING: rewrites history)
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all

   # Force push (if safe to do so)
   git push origin --force --all
   ```

3. **Rotate ALL Credentials** immediately

## Security Checklist

Before deploying to production:

- [ ] `.env` is in `.gitignore`
- [ ] `.env` has never been committed (verify with `git log`)
- [ ] `.env.example` contains only placeholder values
- [ ] Production credentials are stored in secure secrets manager
- [ ] Spending limit is configured appropriately
- [ ] API logs are monitored
- [ ] Twilio webhook URL uses HTTPS
- [ ] (Optional) Request signature validation enabled
- [ ] (Optional) Rate limiting configured
- [ ] Backup/restore procedures documented

## Contact

For security concerns or to report vulnerabilities:
- Create a private issue in GitHub
- Or contact the maintainer directly

---

**Last Updated**: January 2026
**Security Review**: Recommended quarterly
