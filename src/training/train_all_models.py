"""
Complete Model Training & Evaluation Pipeline
Tests all models and selects the best performer

Models evaluated:
1. Heuristic (baseline)
2. AventIQ SMS Spam
3. BERT-tiny SMS Spam
4. URLBert (URL analysis)
5. Custom DistilBERT (fine-tuned on our data)
6. Ensemble models (if needed)
7. Smart Router (best latency/accuracy)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.heuristic_model import HeuristicSpamDetector
from models.nlp_models import (
    AventIQModel, BertTinyModel, URLBertModel,
    CustomDistilBERTModel, EnsembleModel, SmartRouter
)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.load_data()
    
    def load_data(self):
        """Load train/val/test splits"""
        print("="*80)
        print("LOADING DATASETS")
        print("="*80)
        
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.val_df = pd.read_csv(self.data_dir / "val.csv")
        self.test_df = pd.read_csv(self.data_dir / "test.csv")
        
        print(f"Train: {len(self.train_df)} messages")
        print(f"Val:   {len(self.val_df)} messages")
        print(f"Test:  {len(self.test_df)} messages")
        
        # Check for A2P messages
        if 'a2p_type' in self.train_df.columns:
            a2p_count = self.train_df['a2p_type'].notna().sum()
            print(f"\nA2P messages in training: {a2p_count}")
            if a2p_count > 0:
                print("  Types:", self.train_df['a2p_type'].value_counts().to_dict())
    
    def evaluate_model(self, model, model_name: str, test_df: pd.DataFrame = None,
                      measure_latency: bool = True) -> dict:
        """
        Evaluate a model on test set
        
        Returns dict with metrics:
        - accuracy, precision, recall, f1
        - confusion matrix
        - latency (p50, p95, p99)
        - false positive rate on A2P
        """
        if test_df is None:
            test_df = self.test_df
        
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*80}")
        
        messages = test_df['message'].tolist()
        true_labels = test_df['label'].tolist()
        
        # Predict
        predictions = []
        confidences = []
        latencies = []
        
        print(f"Running inference on {len(messages)} messages...")
        
        for i, msg in enumerate(messages):
            start_time = time.time()
            
            try:
                # Handle different model return formats
                result = model.predict(msg)
                if len(result) == 3:
                    # Heuristic model: (label, confidence, details)
                    label, conf, _ = result
                else:
                    # NLP models: (label, confidence)
                    label, conf = result
                
                predictions.append(label)
                confidences.append(conf)
                
                if measure_latency:
                    latency = (time.time() - start_time) * 1000  # ms
                    latencies.append(latency)
                
            except Exception as e:
                print(f"  Error on message {i}: {e}")
                predictions.append("ham")  # Default to ham on error
                confidences.append(0.5)
                latencies.append(0)
            
            # Progress
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(messages)}...")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, pos_label='spam', zero_division=0)
        recall = recall_score(true_labels, predictions, pos_label='spam', zero_division=0)
        f1 = f1_score(true_labels, predictions, pos_label='spam', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=['ham', 'spam'])
        tn, fp, fn, tp = cm.ravel()
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Calculate FPR specifically on A2P messages (if available)
        a2p_fpr = None
        if 'a2p_type' in test_df.columns:
            a2p_mask = test_df['a2p_type'].notna()
            if a2p_mask.sum() > 0:
                a2p_true = test_df[a2p_mask]['label'].tolist()
                a2p_pred = [predictions[i] for i in range(len(predictions)) if a2p_mask.iloc[i]]
                
                # Count false positives on A2P
                a2p_fp = sum(1 for t, p in zip(a2p_true, a2p_pred) if t == 'ham' and p == 'spam')
                a2p_total = len(a2p_true)
                a2p_fpr = a2p_fp / a2p_total if a2p_total > 0 else 0
        
        # Latency statistics
        if latencies:
            latency_p50 = np.percentile(latencies, 50)
            latency_p95 = np.percentile(latencies, 95)
            latency_p99 = np.percentile(latencies, 99)
            avg_latency = np.mean(latencies)
        else:
            latency_p50 = latency_p95 = latency_p99 = avg_latency = 0
        
        # Print results
        print(f"\n{'─'*80}")
        print(f"RESULTS: {model_name}")
        print(f"{'─'*80}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"               Predicted")
        print(f"             Ham    Spam")
        print(f"Actual Ham   {tn:4d}   {fp:4d}  (FPR: {fpr:.4f})")
        print(f"Actual Spam  {fn:4d}   {tp:4d}")
        
        if a2p_fpr is not None:
            print(f"\n⚠️  False Positive Rate on A2P (2FA/Marketing/Transactional): {a2p_fpr:.4f} ({a2p_fpr*100:.2f}%)")
            if a2p_fpr > 0.01:
                print(f"   WARNING: A2P FPR exceeds 1% target!")
        
        if latencies:
            print(f"\nLatency Statistics:")
            print(f"  Average:  {avg_latency:.2f} ms")
            print(f"  p50:      {latency_p50:.2f} ms")
            print(f"  p95:      {latency_p95:.2f} ms")
            print(f"  p99:      {latency_p99:.2f} ms")
            
            if latency_p95 > 50:
                print(f"   WARNING: p95 latency exceeds 50ms target!")
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'fpr': float(fpr),
            'a2p_fpr': float(a2p_fpr) if a2p_fpr is not None else None,
            'confusion_matrix': cm.tolist(),
            'latency': {
                'avg': float(avg_latency),
                'p50': float(latency_p50),
                'p95': float(latency_p95),
                'p99': float(latency_p99),
            },
            'meets_targets': {
                'f1': f1 >= 0.96,
                'a2p_fpr': a2p_fpr < 0.01 if a2p_fpr is not None else True,
                'latency': latency_p95 < 50 if latencies else True
            }
        }
        
        self.results[model_name] = results
        return results


def main():
    """Main training and evaluation pipeline"""
    
    print("="*80)
    print("SMS SPAM DETECTION - MODEL TRAINING & EVALUATION")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Phase 1: Evaluate Heuristic Model (Baseline)
    print("\n" + "="*80)
    print("PHASE 1: HEURISTIC MODEL (BASELINE)")
    print("="*80)
    
    heuristic_model = HeuristicSpamDetector()
    heuristic_results = evaluator.evaluate_model(
        heuristic_model,
        "Heuristic (Text + URL)"
    )
    
    # Phase 2: Evaluate Pre-trained HuggingFace Models
    print("\n" + "="*80)
    print("PHASE 2: PRE-TRAINED HUGGINGFACE MODELS")
    print("="*80)
    
    # 2a. AventIQ Model
    print("\n→ Testing AventIQ SMS Spam Detection Model...")
    try:
        aventiq_model = AventIQModel()
        aventiq_model.load_model()
        aventiq_results = evaluator.evaluate_model(
            aventiq_model,
            "AventIQ SMS Spam"
        )
    except Exception as e:
        print(f"❌ AventIQ model failed: {e}")
        aventiq_results = None
    
    # 2b. BERT-tiny Model
    print("\n→ Testing BERT-tiny SMS Spam Model...")
    try:
        bert_tiny_model = BertTinyModel()
        bert_tiny_model.load_model()
        bert_tiny_results = evaluator.evaluate_model(
            bert_tiny_model,
            "BERT-tiny SMS Spam"
        )
    except Exception as e:
        print(f"❌ BERT-tiny model failed: {e}")
        bert_tiny_results = None
    
    # Phase 3: Train Custom DistilBERT
    print("\n" + "="*80)
    print("PHASE 3: CUSTOM DISTILBERT (FINE-TUNED)")
    print("="*80)
    
    custom_results = None
    try:
        # Prepare data for training
        train_texts = evaluator.train_df['message'].tolist()
        train_labels = [1 if label == 'spam' else 0 for label in evaluator.train_df['label']]
        val_texts = evaluator.val_df['message'].tolist()
        val_labels = [1 if label == 'spam' else 0 for label in evaluator.val_df['label']]
        
        # Train
        custom_model = CustomDistilBERTModel()
        output_dir = "models/distilbert_spam"
        
        print("\nStarting DistilBERT fine-tuning...")
        print("This may take 10-30 minutes depending on your hardware...")
        
        custom_model.train(
            train_texts, train_labels,
            val_texts, val_labels,
            output_dir
        )
        
        # Evaluate
        custom_model.load_model(output_dir)
        custom_results = evaluator.evaluate_model(
            custom_model,
            "Custom DistilBERT"
        )
        
    except Exception as e:
        print(f"❌ Custom DistilBERT training failed: {e}")
        print("   Continuing with pre-trained models only...")
    
    # Phase 4: Create Ensemble (if multiple models meet targets)
    print("\n" + "="*80)
    print("PHASE 4: ENSEMBLE MODELS")
    print("="*80)
    
    # Collect models that meet F1 target
    good_models = []
    model_weights = []
    
    for name, results in evaluator.results.items():
        if results and results['f1_score'] >= 0.90:  # Lower threshold for ensemble candidates
            print(f"✅ {name}: F1={results['f1_score']:.4f} (qualified for ensemble)")
            
            # Get model object
            if name == "Heuristic (Text + URL)":
                good_models.append(heuristic_model)
                model_weights.append(results['f1_score'])
            elif name == "AventIQ SMS Spam" and aventiq_results:
                good_models.append(aventiq_model)
                model_weights.append(results['f1_score'])
            elif name == "BERT-tiny SMS Spam" and bert_tiny_results:
                good_models.append(bert_tiny_model)
                model_weights.append(results['f1_score'])
            elif name == "Custom DistilBERT" and custom_results:
                good_models.append(custom_model)
                model_weights.append(results['f1_score'])
    
    ensemble_results = None
    if len(good_models) >= 2:
        print(f"\nCreating weighted ensemble with {len(good_models)} models...")
        ensemble = EnsembleModel(good_models, weights=model_weights, strategy="weighted")
        ensemble_results = evaluator.evaluate_model(
            ensemble,
            f"Ensemble ({len(good_models)} models)"
        )
    else:
        print("\nℹ️  Not enough models for ensemble (need 2+)")
    
    # Phase 5: Smart Router
    print("\n" + "="*80)
    print("PHASE 5: SMART ROUTER (OPTIMIZED LATENCY)")
    print("="*80)
    
    # Use heuristic as fast model, best performing as slow model
    best_model_name = max(evaluator.results.items(), key=lambda x: x[1]['f1_score'])[0]
    print(f"\nCreating Smart Router:")
    print(f"  Fast path: Heuristic (Text + URL)")
    print(f"  Slow path: {best_model_name}")
    
    # Get best model object
    if best_model_name == "AventIQ SMS Spam" and aventiq_results:
        best_model = aventiq_model
    elif best_model_name == "BERT-tiny SMS Spam" and bert_tiny_results:
        best_model = bert_tiny_model
    elif best_model_name == "Custom DistilBERT" and custom_results:
        best_model = custom_model
    elif best_model_name.startswith("Ensemble") and ensemble_results:
        best_model = ensemble
    else:
        best_model = heuristic_model
    
    smart_router = SmartRouter(
        fast_model=heuristic_model,
        slow_model=best_model,
        threshold=0.85  # High confidence threshold
    )
    
    router_results = evaluator.evaluate_model(
        smart_router,
        "Smart Router"
    )
    
    # Print routing stats
    routing_stats = smart_router.get_routing_stats()
    print(f"\nRouting Statistics:")
    print(f"  Fast path: {routing_stats['fast_count']} ({routing_stats['fast_percentage']:.1f}%)")
    print(f"  Slow path: {routing_stats['slow_count']} ({100-routing_stats['fast_percentage']:.1f}%)")
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<30} {'F1':>8} {'A2P FPR':>10} {'Latency':>12} {'Meets Targets':>15}")
    print("─" * 80)
    
    for name, results in sorted(evaluator.results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
        f1 = results['f1_score']
        a2p_fpr = results['a2p_fpr']
        latency = results['latency']['p95']
        
        a2p_str = f"{a2p_fpr*100:.2f}%" if a2p_fpr is not None else "N/A"
        latency_str = f"{latency:.1f} ms"
        
        meets_all = all(results['meets_targets'].values())
        status = "✅ YES" if meets_all else "❌ NO"
        
        print(f"{name:<30} {f1:>8.4f} {a2p_str:>10} {latency_str:>12} {status:>15}")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Find best model that meets all targets
    qualifying_models = {
        name: results for name, results in evaluator.results.items()
        if all(results['meets_targets'].values())
    }
    
    if qualifying_models:
        # Choose model with best F1 among qualifying models
        best_name = max(qualifying_models.items(), key=lambda x: x[1]['f1_score'])[0]
        best = qualifying_models[best_name]
        
        print(f"\n🎯 RECOMMENDED MODEL: {best_name}")
        print(f"\n   Performance:")
        print(f"   - F1 Score: {best['f1_score']:.4f} (target: ≥0.96) {'✅' if best['f1_score'] >= 0.96 else '⚠️'}")
        print(f"   - A2P FPR: {best['a2p_fpr']*100:.2f}% (target: <1.0%) {'✅' if best['a2p_fpr'] < 0.01 else '⚠️'}")
        print(f"   - Latency p95: {best['latency']['p95']:.1f}ms (target: <50ms) {'✅' if best['latency']['p95'] < 50 else '⚠️'}")
        
        print(f"\n   This model achieves the best balance of:")
        print(f"   - High accuracy for spam/smishing detection")
        print(f"   - Low false positives on legitimate A2P traffic")
        print(f"   - Fast inference for real-time use")
    else:
        print("\n⚠️  No model meets all targets. Consider:")
        print("   1. Collecting more training data (especially A2P examples)")
        print("   2. Adjusting classification threshold")
        print("   3. Using ensemble with adjusted weights")
        
        # Recommend best by F1 score
        best_name = max(evaluator.results.items(), key=lambda x: x[1]['f1_score'])[0]
        print(f"\n   Best F1 score: {best_name}")
    
    # Save all results
    results_file = Path("models/evaluation_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'results': evaluator.results,
            'recommendation': best_name if qualifying_models else None
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    print(f"\nNext step: Deploy recommended model with FastAPI")
    print(f"           python src/api/main.py")


if __name__ == "__main__":
    main()
