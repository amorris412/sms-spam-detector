"""
Simple API starter that loads Custom DistilBERT directly
"""
import sys
from pathlib import Path
sys.path.append('src')

from fastapi import FastAPI
from models.nlp_models import CustomDistilBERTModel

# Initialize
print("Loading Custom DistilBERT model...")
model = CustomDistilBERTModel()
model.load_model("models/distilbert_spam")
print("✅ Model loaded!")

# Now start the API
import uvicorn
uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
