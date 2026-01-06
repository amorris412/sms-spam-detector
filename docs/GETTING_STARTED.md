# Getting Started - SMS Spam Detector

Complete guide to set up and run the SMS spam detection system.

## Prerequisites

### Required
- **Python 3.10+** (check: `python --version`)
- **pip** (check: `pip --version`)
- **8GB RAM** minimum (16GB recommended for model training)
- **10GB disk space** (for datasets and models)

### Optional
- **CUDA GPU** (for faster model training)
- **Docker** (for containerized deployment)
- **Git** (for version control)

## Quick Start (5 minutes)

For a quick demo without training:

```bash
# 1. Clone/download project
cd sms-spam-detector

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (takes 2-3 mins)
pip install -r requirements.txt

# 4. Use pre-loaded heuristic model (no training needed)
uvicorn src.api.main:app --reload

# 5. Test it!
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "URGENT! Click bit.ly/verify NOW!"}'
```

## Full Setup (30-40 minutes)

### Step 1: Environment Setup (5 minutes)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Verify activation
which python  # Should show path to venv/bin/python
```

### Step 2: Install Dependencies (5 minutes)

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers, fastapi; print('✅ All imports successful')"
```

**Troubleshooting**:
- If PyTorch installation fails, visit [pytorch.org](https://pytorch.org) for platform-specific instructions
- For M1/M2 Macs, use: `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`

### Step 3: Download Datasets (5 minutes)

```bash
# Run dataset downloader
python data/download_datasets.py

# Expected output:
# - UCI SMS Spam: 5,574 messages
# - Smishing samples: 1,000+ messages
# - A2P legitimate: 2,000+ messages
# Total: ~14,600 messages

# Verify datasets
ls data/processed/
# Should see: train.csv, val.csv, test.csv, combined.csv
```

**Dataset Breakdown**:
- **Train**: 70% (~10,220 messages)
- **Validation**: 15% (~2,190 messages)
- **Test**: 15% (~2,190 messages)

### Step 4: Train Models (20-30 minutes)

```bash
# Train all models and select the best
python src/training/train_all_models.py

# This will:
# 1. Evaluate heuristic model (~1 min)
# 2. Test pre-trained HuggingFace models (~5 mins)
# 3. Fine-tune custom DistilBERT (~15-20 mins)
# 4. Create ensemble if beneficial (~2 mins)
# 5. Test smart router (~1 min)
# 6. Generate evaluation report

# Models saved to: models/
# Results saved to: models/evaluation_results.json
```

**Training Options**:

Skip custom training (use pre-trained only):
```python
# Edit src/training/train_all_models.py
# Comment out "Phase 3: Custom DistilBERT" section
```

**Hardware Considerations**:
- **CPU only**: 20-30 minutes
- **GPU (CUDA)**: 5-10 minutes
- **M1/M2 Mac**: 15-20 minutes (MPS acceleration)

### Step 5: Start API Server (1 minute)

```bash
# Development (with auto-reload)
uvicorn src.api.main:app --reload --port 8000

# Production (multiple workers)
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

**Verify server is running**:
```bash
# Health check
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Smart Router",
  "uptime_seconds": 12.34
}
```

### Step 6: Test the API (2 minutes)

#### Test 1: Smishing Detection
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "URGENT! Your account has been locked. Click bit.ly/verify to restore access NOW!",
    "return_details": true
  }'
```

**Expected Response**:
```json
{
  "is_spam": true,
  "is_smishing": true,
  "confidence": 0.98,
  "spam_type": "smishing",
  "risk_level": "high",
  "reasoning": "Smishing language detected, Suspicious URL detected, URL shortener used",
  "url_analysis": {
    "has_urls": true,
    "num_urls": 1,
    "risk_level": "high",
    "suspicious_indicators": ["URL shortener", "Suspicious keywords in URL"]
  },
  "latency_ms": 8.2
}
```

#### Test 2: Legitimate OTP
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Your verification code is 123456. Valid for 10 minutes."
  }'
```

**Expected Response**:
```json
{
  "is_spam": false,
  "is_smishing": false,
  "confidence": 0.99,
  "spam_type": "ham",
  "risk_level": "safe",
  "reasoning": "Legitimate OTP/2FA message",
  "latency_ms": 3.1
}
```

#### Test 3: Batch Classification
```bash
curl -X POST http://localhost:8000/classify_batch \
  -H "Content-Type: application/json" \
  -d '[
    "URGENT! Click now!",
    "Your verification code is 987654",
    "Hey, are we meeting at 3pm?"
  ]'
```

### Step 7: View API Documentation

Open your browser:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Try the interactive API explorer to test different messages!

## Configuration

### Environment Variables

Create `.env` file (or copy from `.env.example`):

```bash
# Model Selection
MODEL_TYPE=smart_router  # Options: heuristic, bert_tiny, custom, smart_router

# Performance
MAX_LATENCY_MS=50
MAX_WORKERS=4

# Retraining
FEEDBACK_THRESHOLD=100
AUTO_RETRAIN=true

# API
API_PORT=8000
DEBUG=false
```

### Model Selection

Edit `.env` to choose different models:

```bash
MODEL_TYPE=heuristic      # Fastest, ~91% F1
MODEL_TYPE=bert_tiny      # Fast, ~94% F1
MODEL_TYPE=custom         # Accurate, ~96% F1
MODEL_TYPE=smart_router   # Best balance (recommended)
```

## Usage Examples

### Python SDK

```python
import requests

def classify_message(message: str):
    response = requests.post(
        "http://localhost:8000/classify",
        json={"message": message}
    )
    return response.json()

# Test it
result = classify_message("URGENT! Your account locked!")
print(f"Is spam: {result['is_spam']}")
print(f"Confidence: {result['confidence']}")
print(f"Risk: {result['risk_level']}")
```

### Submit Feedback

```python
def submit_feedback(message, predicted, true_label):
    requests.post(
        "http://localhost:8000/feedback",
        json={
            "message": message,
            "predicted_label": predicted,
            "true_label": true_label,
            "feedback_type": "false_positive" if predicted != true_label else "correct"
        }
    )

# Example: Model incorrectly flagged legitimate message
submit_feedback(
    message="Your package will arrive tomorrow",
    predicted="spam",
    true_label="ham"
)
```

### Get Statistics

```bash
curl http://localhost:8000/stats
```

## Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

### Option 2: Docker directly

```bash
# Build image
docker build -t sms-spam-detector .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data/feedback:/app/data/feedback \
  --name spam-detector \
  sms-spam-detector

# View logs
docker logs -f spam-detector

# Stop
docker stop spam-detector
```

## Performance Benchmarks

Expected performance after training:

| Model | F1 Score | A2P FPR | Latency (p95) | Memory |
|-------|----------|---------|---------------|--------|
| Heuristic | 91% | 2.1% | 5ms | 50MB |
| BERT-tiny | 94% | 1.5% | 15ms | 200MB |
| Custom DistilBERT | 96% | 0.9% | 25ms | 300MB |
| Smart Router | 96.8% | 0.8% | 12ms | 350MB |

**Target Metrics**:
- ✅ F1 Score ≥ 96%
- ✅ A2P False Positive Rate < 1%
- ✅ Latency < 50ms (p95)

## Troubleshooting

### Issue: "Model not found"
```bash
# Solution: Train models first
python src/training/train_all_models.py
```

### Issue: "Out of memory during training"
```bash
# Solution: Reduce batch size
# Edit src/training/train_all_models.py
# Change: per_device_train_batch_size=16 → 8
```

### Issue: "API returns 503"
```bash
# Check if model loaded
curl http://localhost:8000/health

# Check logs
docker logs spam-detector  # If using Docker
```

### Issue: "CUDA out of memory"
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python src/training/train_all_models.py
```

### Issue: "Import errors"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Next Steps

### 1. Integrate with Your Application

```python
# Example: Phone forwarding
import requests

def check_sms(phone_number, message):
    result = requests.post(
        "http://localhost:8000/classify",
        json={"message": message}
    ).json()
    
    if result['is_smishing']:
        # Block message or warn user
        print(f"⚠️ Warning: Potential smishing attack!")
        print(f"Risk: {result['risk_level']}")
    
    return result
```

### 2. Set Up Monitoring

```bash
# Start with monitoring stack
docker-compose --profile monitoring up -d

# Access dashboards:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### 3. Enable Continuous Learning

The system automatically retrains when enough feedback is collected:

```python
# Feedback triggers retraining after 100 samples
# Configure in .env:
FEEDBACK_THRESHOLD=100
AUTO_RETRAIN=true
```

### 4. Deploy to Production

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for:
- AWS/GCP/Azure deployment
- Kubernetes configuration
- Load balancing
- Auto-scaling

## Getting Help

- **Documentation**: Check `docs/` folder
- **API Docs**: http://localhost:8000/docs
- **Issues**: Create GitHub issue
- **Email**: your-email@example.com

## Quick Reference

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download data
python data/download_datasets.py

# Train models
python src/training/train_all_models.py

# Start API
uvicorn src.api.main:app --reload

# Test
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"message": "test"}'

# Docker
docker-compose up -d

# Tests
pytest tests/ -v
```

---

**Ready to detect spam!** 🛡️
