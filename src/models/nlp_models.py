"""
NLP Models for SMS Spam Detection
Integrates multiple HuggingFace models:
1. AventIQ SMS Spam Detection
2. BERT-tiny SMS Spam
3. URLBert Phishing Classifier
4. Custom fine-tuned DistilBERT

Also includes ensemble strategies
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
import numpy as np
from typing import Dict, List, Tuple
import time
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from features.url_features import URLFeatureExtractor


class NLPModelWrapper:
    """Base wrapper for HuggingFace models"""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self):
        """Load model and tokenizer"""
        raise NotImplementedError
        
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict spam/ham for a single text"""
        raise NotImplementedError
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict for batch of texts"""
        raise NotImplementedError


class AventIQModel(NLPModelWrapper):
    """AventIQ SMS Spam Detection Model"""
    
    def __init__(self):
        super().__init__("AventIQ-AI/SMS-Spam-Detection-Model")
        
    def load_model(self):
        """Load AventIQ model"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict spam/ham"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
        # Get prediction
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        # Assuming 0=ham, 1=spam (verify with model card)
        label = "spam" if pred_idx == 1 else "ham"
        
        return label, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch prediction"""
        results = []
        for text in texts:
            label, conf = self.predict(text)
            results.append((label, conf))
        return results


class BertTinyModel(NLPModelWrapper):
    """BERT-tiny SMS Spam Model"""
    
    def __init__(self):
        super().__init__("mrm8488/bert-tiny-finetuned-sms-spam-detection")
    
    def load_model(self):
        """Load BERT-tiny model"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict spam/ham"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        # BERT-tiny: 0=ham, 1=spam
        label = "spam" if pred_idx == 1 else "ham"
        
        return label, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch prediction"""
        results = []
        for text in texts:
            label, conf = self.predict(text)
            results.append((label, conf))
        return results


class URLBertModel(NLPModelWrapper):
    """URLBert Phishing Classifier (for URLs only)"""
    
    def __init__(self):
        super().__init__("CrabInHoney/urlbert-tiny-v4-phishing-classifier")
        self.url_extractor = URLFeatureExtractor()
    
    def load_model(self):
        """Load URLBert model"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
    
    def predict_url(self, url: str) -> Tuple[str, float]:
        """Predict if URL is phishing"""
        inputs = self.tokenizer(url, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        # Assuming 0=benign, 1=phishing
        label = "phishing" if pred_idx == 1 else "benign"
        
        return label, confidence
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Analyze URLs in message text
        Returns 'spam' if any URL is phishing, else 'ham'
        """
        urls = self.url_extractor.extract_urls(text)
        
        if not urls:
            return "ham", 0.5  # No URLs to analyze
        
        # Check all URLs
        phishing_count = 0
        max_confidence = 0.0
        
        for url in urls:
            label, conf = self.predict_url(url)
            if label == "phishing":
                phishing_count += 1
                max_confidence = max(max_confidence, conf)
        
        if phishing_count > 0:
            return "spam", max_confidence
        else:
            return "ham", 0.7
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch prediction"""
        results = []
        for text in texts:
            label, conf = self.predict(text)
            results.append((label, conf))
        return results


class CustomDistilBERTModel:
    """
    Custom DistilBERT fine-tuned on our combined dataset
    Provides best performance for our specific use case
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/distilbert_spam"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
    
    def train(self, train_texts, train_labels, val_texts, val_labels, output_dir):
        """
        Fine-tune DistilBERT on training data
        
        Args:
            train_texts: List of training messages
            train_labels: List of training labels (0=ham, 1=spam)
            val_texts: List of validation messages
            val_labels: List of validation labels
            output_dir: Where to save the model
        """
        print("Training Custom DistilBERT model...")
        
        # Load pretrained model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        
        # Tokenize datasets
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Create PyTorch datasets
        class SMSDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = SMSDataset(train_encodings, train_labels)
        val_dataset = SMSDataset(val_encodings, val_labels)
        
        # Training arguments (updated for newer transformers)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model saved to {output_dir}")
    
    def load_model(self, model_path: str = None):
        """Load trained model"""
        path = model_path or self.model_path
        print(f"Loading custom DistilBERT from {path}...")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        self.model = DistilBertForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict spam/ham"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        # 0=ham, 1=spam
        label = "spam" if pred_idx == 1 else "ham"
        
        return label, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch prediction"""
        results = []
        for text in texts:
            label, conf = self.predict(text)
            results.append((label, conf))
        return results


class EnsembleModel:
    """
    Ensemble multiple models for improved accuracy
    Strategies: Voting, Weighted, Stacked
    """
    
    def __init__(self, models: List, weights: List[float] = None, strategy: str = "weighted"):
        """
        Args:
            models: List of model objects
            weights: List of weights for each model (for weighted ensemble)
            strategy: 'voting', 'weighted', or 'stacked'
        """
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.strategy = strategy
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Ensemble prediction"""
        # Get predictions from all models
        predictions = []
        confidences = []
        
        for model in self.models:
            label, conf = model.predict(text)
            predictions.append(1 if label == "spam" else 0)
            confidences.append(conf if label == "spam" else 1 - conf)
        
        if self.strategy == "voting":
            # Simple majority vote
            spam_votes = sum(predictions)
            final_label = "spam" if spam_votes > len(predictions) / 2 else "ham"
            final_conf = spam_votes / len(predictions)
        
        elif self.strategy == "weighted":
            # Weighted average of confidences
            weighted_score = sum(
                p * c * w 
                for p, c, w in zip(predictions, confidences, self.weights)
            )
            final_label = "spam" if weighted_score > 0.5 else "ham"
            final_conf = weighted_score if final_label == "spam" else 1 - weighted_score
        
        else:
            # Default to weighted
            weighted_score = sum(
                p * c * w 
                for p, c, w in zip(predictions, confidences, self.weights)
            )
            final_label = "spam" if weighted_score > 0.5 else "ham"
            final_conf = weighted_score if final_label == "spam" else 1 - weighted_score
        
        return final_label, final_conf
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch prediction"""
        results = []
        for text in texts:
            label, conf = self.predict(text)
            results.append((label, conf))
        return results


class SmartRouter:
    """
    Smart routing: Use fast heuristic model for simple cases,
    ensemble for complex cases
    """
    
    def __init__(self, fast_model, slow_model, threshold: float = 0.8):
        """
        Args:
            fast_model: Fast model (heuristic)
            slow_model: Accurate but slower model (NLP or ensemble)
            threshold: Confidence threshold for fast model
        """
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.threshold = threshold
        self.fast_count = 0
        self.slow_count = 0
    
    def predict(self, text: str) -> Tuple[str, float, Dict]:
        """Route prediction to appropriate model"""
        # Try fast model first
        if hasattr(self.fast_model, 'predict'):
            # Heuristic model returns (label, confidence, details)
            label, conf, details = self.fast_model.predict(text)
        else:
            # NLP model returns (label, confidence)
            label, conf = self.fast_model.predict(text)
            details = {}
        
        # If confident, use fast result
        if conf >= self.threshold:
            self.fast_count += 1
            details['routed_to'] = 'fast_model'
            return label, conf, details
        
        # Otherwise, use slow model
        self.slow_count += 1
        label, conf = self.slow_model.predict(text)
        details['routed_to'] = 'slow_model'
        
        return label, conf, details
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics"""
        total = self.fast_count + self.slow_count
        return {
            'fast_count': self.fast_count,
            'slow_count': self.slow_count,
            'fast_percentage': self.fast_count / total * 100 if total > 0 else 0
        }


if __name__ == "__main__":
    # Test model loading
    print("Testing NLP Model Wrappers")
    print("=" * 80)
    
    # Test messages
    test_messages = [
        "URGENT! Click bit.ly/verify NOW!",
        "Your verification code is 123456",
        "Win $5000 now! Text WIN!"
    ]
    
    print("\nTest 1: AventIQ Model")
    try:
        model = AventIQModel()
        model.load_model()
        for msg in test_messages[:1]:
            label, conf = model.predict(msg)
            print(f"  {msg[:50]}... → {label} ({conf:.3f})")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\nTest 2: BERT-tiny Model")
    try:
        model = BertTinyModel()
        model.load_model()
        for msg in test_messages[:1]:
            label, conf = model.predict(msg)
            print(f"  {msg[:50]}... → {label} ({conf:.3f})")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Model wrapper tests complete!")
