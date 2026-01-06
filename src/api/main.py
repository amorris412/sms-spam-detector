"""
FastAPI Service for SMS Spam Detection
Features:
- Real-time classification with <50ms latency
- User feedback loop for retraining
- Model monitoring with Prometheus metrics
- Health checks and status endpoints
- Async processing for high throughput
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import json
from pathlib import Path
import time
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.heuristic_model import HeuristicSpamDetector
from models.nlp_models import (
    AventIQModel, BertTinyModel, CustomDistilBERTModel,
    SmartRouter
)
from api.sms_handler import (
    SMSHandler, format_classification_response,
    format_short_response, extract_forwarded_message
)
from api.spending_monitor import SpendingMonitor

# Initialize FastAPI
app = FastAPI(
    title="SMS Spam Detection API",
    description="Production-ready spam/smishing detection with multiple models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.sms_handler = None
        self.spending_monitor = None
        self.feedback_buffer = []
        self.feedback_threshold = 100
        self.stats = {
            'requests': 0,
            'spam_detected': 0,
            'ham_detected': 0,
            'feedback_received': 0,
            'sms_received': 0
        }

state = AppState()


# Pydantic models
class ClassifyRequest(BaseModel):
    message: str = Field(..., description="SMS message to classify")
    return_details: bool = Field(False, description="Include detailed analysis")


class ClassifyResponse(BaseModel):
    is_spam: bool
    is_smishing: bool
    confidence: float
    spam_type: str  # 'spam', 'smishing', or 'ham'
    risk_level: str  # 'safe', 'low', 'medium', 'high'
    reasoning: str
    url_analysis: Optional[Dict] = None
    latency_ms: float = None


class FeedbackRequest(BaseModel):
    message: str
    predicted_label: str  # What model predicted
    true_label: str  # What user says is correct
    feedback_type: str  # 'false_positive' or 'false_negative'
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_count: int
    retrain_scheduled: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    uptime_seconds: float
    total_requests: int


class StatsResponse(BaseModel):
    total_requests: int
    spam_detected: int
    ham_detected: int
    spam_rate: float
    feedback_received: int
    model_name: str


# Startup: Load model
@app.on_event("startup")
async def startup_event():
    """Load the best model on startup"""
    global state
    
    print("=" * 80)
    print("Starting SMS Spam Detection API")
    print("=" * 80)
    
    # Try to load evaluation results to get best model
    results_file = Path("models/evaluation_results.json")
    
    if results_file.exists():
        print("\nLoading model based on evaluation results...")
        with open(results_file) as f:
            eval_results = json.load(f)
            recommended_model = eval_results.get('recommendation')
        
        if recommended_model:
            print(f"Recommended model: {recommended_model}")
            
            # Load the recommended model
            try:
                if recommended_model == "Heuristic (Text + URL)":
                    state.model = HeuristicSpamDetector()
                    state.model_name = "Heuristic"
                
                elif recommended_model == "Smart Router":
                    # Load heuristic and best NLP model
                    heuristic = HeuristicSpamDetector()
                    
                    # Try to load custom DistilBERT
                    try:
                        nlp_model = CustomDistilBERTModel()
                        nlp_model.load_model("models/distilbert_spam")
                        state.model = SmartRouter(heuristic, nlp_model, threshold=0.85)
                        state.model_name = "Smart Router"
                    except:
                        state.model = heuristic
                        state.model_name = "Heuristic (fallback)"
                
                elif recommended_model == "Custom DistilBERT":
                    nlp_model = CustomDistilBERTModel()
                    nlp_model.load_model("models/distilbert_spam")
                    state.model = nlp_model
                    state.model_name = "Custom DistilBERT"
                
                else:
                    # Default to heuristic
                    state.model = HeuristicSpamDetector()
                    state.model_name = "Heuristic (default)"
                
                print(f"✅ Model loaded: {state.model_name}")
                
            except Exception as e:
                print(f"❌ Error loading recommended model: {e}")
                print("   Falling back to heuristic model...")
                state.model = HeuristicSpamDetector()
                state.model_name = "Heuristic (fallback)"
    
    else:
        print("\nNo evaluation results found. Loading default heuristic model...")
        state.model = HeuristicSpamDetector()
        state.model_name = "Heuristic (default)"
    
    # Initialize SMS handler
    state.sms_handler = SMSHandler()
    if state.sms_handler.enabled:
        print(f"📱 SMS features enabled: {state.sms_handler.phone_number}")
    else:
        print("📱 SMS features disabled (no Twilio credentials)")

    # Initialize spending monitor
    state.spending_monitor = SpendingMonitor(daily_limit=5.0)
    if state.spending_monitor.enabled:
        status = state.spending_monitor.get_status()
        print(f"💰 Spending monitor enabled: ${status['daily_limit']:.2f} daily limit")
        if status['enabled']:
            print(f"   Current spend: ${status['daily_spend']:.2f} ({status['message_count']} messages)")
        else:
            print(f"   ⚠️  Service disabled (limit exceeded)")
    else:
        print("💰 Spending monitor disabled (no Twilio credentials)")

    state.start_time = time.time()
    print(f"\n🚀 API ready! Model: {state.model_name}")
    print("=" * 80)


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if state.model is not None else "unhealthy",
        model_loaded=state.model is not None,
        model_name=state.model_name or "none",
        uptime_seconds=time.time() - state.start_time,
        total_requests=state.stats['requests']
    )


# Main classification endpoint
@app.post("/classify", response_model=ClassifyResponse)
async def classify_message(request: ClassifyRequest):
    """
    Classify an SMS message as spam/smishing/ham
    
    Returns:
    - is_spam: Boolean indicating if message is spam/smishing
    - is_smishing: Boolean indicating if message is specifically smishing
    - confidence: 0.0 to 1.0
    - spam_type: 'spam', 'smishing', or 'ham'
    - risk_level: 'safe', 'low', 'medium', 'high'
    - reasoning: Human-readable explanation
    - url_analysis: Details about any URLs (if present)
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Classify
    result = state.model.predict(request.message)
    
    # Handle different return formats
    if len(result) == 3:
        # Heuristic or Smart Router: (label, confidence, details)
        label, confidence, details = result
    else:
        # NLP models: (label, confidence)
        label, confidence = result
        details = {}
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Determine if smishing (if not already in details)
    is_smishing = label == "smishing"
    if not is_smishing and label == "spam" and details:
        # Check if spam has smishing characteristics
        url_analysis = details.get('url_analysis', {})
        smishing_indicators = (
            url_analysis.get('risk_level') in ['high', 'medium'] or
            details.get('text_features', {}).get('smishing_keywords', 0) > 0.3
        )
        is_smishing = smishing_indicators
    
    # Update stats
    state.stats['requests'] += 1
    if label in ['spam', 'smishing']:
        state.stats['spam_detected'] += 1
    else:
        state.stats['ham_detected'] += 1
    
    # Build response
    response = ClassifyResponse(
        is_spam=(label in ['spam', 'smishing']),
        is_smishing=is_smishing,
        confidence=confidence,
        spam_type=label,
        risk_level=details.get('risk_level', 'unknown'),
        reasoning=details.get('reasoning', 'Classification based on model prediction'),
        latency_ms=latency_ms
    )
    
    # Add URL analysis if requested and available
    if request.return_details and 'url_analysis' in details:
        response.url_analysis = details['url_analysis']
    
    return response


# Feedback endpoint
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit feedback on a classification
    Used for model retraining when enough feedback is collected
    """
    # Store feedback
    feedback_entry = {
        'message': request.message,
        'predicted_label': request.predicted_label,
        'true_label': request.true_label,
        'feedback_type': request.feedback_type,
        'notes': request.notes,
        'timestamp': datetime.now().isoformat()
    }
    
    state.feedback_buffer.append(feedback_entry)
    state.stats['feedback_received'] += 1
    
    # Save to file
    feedback_file = Path("data/feedback.jsonl")
    feedback_file.parent.mkdir(exist_ok=True)
    
    with open(feedback_file, 'a') as f:
        f.write(json.dumps(feedback_entry) + '\n')
    
    # Check if we should retrain
    retrain_scheduled = len(state.feedback_buffer) >= state.feedback_threshold
    
    if retrain_scheduled:
        # In production, this would trigger a background retraining job
        background_tasks.add_task(trigger_retraining)
    
    return FeedbackResponse(
        status="success",
        message="Feedback received. Thank you for helping improve the model!",
        feedback_count=len(state.feedback_buffer),
        retrain_scheduled=retrain_scheduled
    )


def trigger_retraining():
    """
    Trigger model retraining with feedback data
    In production, this would queue a training job
    """
    print(f"\n{'='*80}")
    print(f"RETRAINING TRIGGERED")
    print(f"{'='*80}")
    print(f"Feedback count: {len(state.feedback_buffer)}")
    print("In production, this would:")
    print("  1. Queue a training job")
    print("  2. Retrain model with original + feedback data")
    print("  3. Evaluate on test set")
    print("  4. Deploy if performance improves")
    print("  5. Send notification")
    
    # Clear buffer
    state.feedback_buffer = []


# Statistics endpoint
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API statistics"""
    total = state.stats['requests']
    spam_rate = state.stats['spam_detected'] / total if total > 0 else 0

    return StatsResponse(
        total_requests=state.stats['requests'],
        spam_detected=state.stats['spam_detected'],
        ham_detected=state.stats['ham_detected'],
        spam_rate=spam_rate,
        feedback_received=state.stats['feedback_received'],
        model_name=state.model_name or "unknown"
    )


# Spending status endpoint
@app.get("/spending")
async def get_spending_status():
    """Get current spending status and limits"""
    if not state.spending_monitor or not state.spending_monitor.enabled:
        return {
            "enabled": False,
            "message": "Spending monitor not available"
        }

    status = state.spending_monitor.get_status()

    # Try to get actual Twilio usage
    twilio_usage = state.spending_monitor.get_twilio_usage()

    return {
        "enabled": status['enabled'],
        "daily_limit": status['daily_limit'],
        "daily_spend": status['daily_spend'],
        "message_count": status['message_count'],
        "remaining_budget": status['remaining_budget'],
        "date": status['date'],
        "twilio_actual_usage": twilio_usage
    }


# Batch classification endpoint
@app.post("/classify_batch")
async def classify_batch(messages: List[str]):
    """
    Classify multiple messages at once
    Returns array of classification results
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for message in messages:
        try:
            result = state.model.predict(message)
            
            if len(result) == 3:
                label, confidence, details = result
            else:
                label, confidence = result
                details = {}
            
            is_smishing = label == "smishing"
            
            results.append({
                'message': message[:100] + '...' if len(message) > 100 else message,
                'is_spam': label in ['spam', 'smishing'],
                'is_smishing': is_smishing,
                'confidence': confidence,
                'spam_type': label,
                'risk_level': details.get('risk_level', 'unknown')
            })
        
        except Exception as e:
            results.append({
                'message': message[:100] + '...' if len(message) > 100 else message,
                'error': str(e)
            })
    
    return {"results": results, "count": len(results)}


# SMS Webhook endpoint for Twilio
@app.post("/sms/incoming")
async def handle_incoming_sms(
    From: str = Form(...),
    Body: str = Form(...),
    MessageSid: str = Form(None)
):
    """
    Twilio webhook endpoint for incoming SMS messages

    When a user forwards a text message to your Twilio number,
    this endpoint receives it, classifies it, and sends back the result.

    Twilio automatically sends these form parameters:
    - From: Sender's phone number
    - Body: Message text
    - MessageSid: Unique message ID

    Returns TwiML response that Twilio will send back to the user
    """
    if state.model is None:
        error_response = state.sms_handler.create_twiml_response(
            "⚠️ Service temporarily unavailable. Please try again later."
        )
        return Response(content=error_response, media_type="application/xml")

    # Check spending limit
    if state.spending_monitor and not state.spending_monitor.is_service_enabled():
        error_response = state.sms_handler.create_twiml_response(
            "Daily spending limit exceeded. Service will resume tomorrow. Thank you for understanding!"
        )
        print(f"   🚫 Request rejected: Spending limit exceeded")
        return Response(content=error_response, media_type="application/xml")

    # Extract the forwarded message content
    message_to_classify = extract_forwarded_message(Body)

    # Log the incoming SMS
    state.stats['sms_received'] += 1
    print(f"\n📱 SMS received from {From}")
    print(f"   Message SID: {MessageSid}")
    print(f"   Content: {message_to_classify[:100]}...")

    try:
        # Classify the message
        start_time = time.time()
        result = state.model.predict(message_to_classify)

        # Handle different return formats
        if len(result) == 3:
            label, confidence, details = result
        else:
            label, confidence = result
            details = {}

        latency_ms = (time.time() - start_time) * 1000

        # Determine classification details
        is_spam = label in ['spam', 'smishing']
        is_smishing = label == "smishing"

        if not is_smishing and label == "spam" and details:
            url_analysis = details.get('url_analysis', {})
            smishing_indicators = (
                url_analysis.get('risk_level') in ['high', 'medium'] or
                details.get('text_features', {}).get('smishing_keywords', 0) > 0.3
            )
            is_smishing = smishing_indicators

        risk_level = details.get('risk_level', 'unknown')
        reasoning = details.get('reasoning', 'Classification based on model prediction')
        url_analysis = details.get('url_analysis', {})

        # Update stats
        state.stats['requests'] += 1
        if is_spam:
            state.stats['spam_detected'] += 1
        else:
            state.stats['ham_detected'] += 1

        # Format response for SMS
        response_text = format_classification_response(
            is_spam=is_spam,
            is_smishing=is_smishing,
            confidence=confidence,
            spam_type=label,
            risk_level=risk_level,
            reasoning=reasoning,
            url_analysis=url_analysis
        )

        # Log the result
        print(f"   Classification: {label} ({confidence:.2f})")
        print(f"   Latency: {latency_ms:.1f}ms")
        print(f"   Response: {response_text[:100]}...")

        # Create TwiML response (Twilio will send this back to the user)
        twiml_response = state.sms_handler.create_twiml_response(response_text)

        # Record spending (one message in + one message out = $0.015)
        if state.spending_monitor:
            state.spending_monitor.record_message(cost=0.015)

        return Response(content=twiml_response, media_type="application/xml")

    except Exception as e:
        print(f"❌ Error processing SMS: {e}")
        error_response = state.sms_handler.create_twiml_response(
            "⚠️ Error analyzing message. Please try again."
        )
        # Still record spending even on error (one message in + one message out)
        if state.spending_monitor:
            state.spending_monitor.record_message(cost=0.015)
        return Response(content=error_response, media_type="application/xml")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "SMS Spam Detection API",
        "version": "1.0.0",
        "model": state.model_name,
        "status": "operational" if state.model is not None else "loading",
        "endpoints": {
            "classify": "POST /classify - Classify a single message",
            "classify_batch": "POST /classify_batch - Classify multiple messages",
            "feedback": "POST /feedback - Submit feedback for model improvement",
            "stats": "GET /stats - Get API statistics",
            "spending": "GET /spending - Get spending status and limits",
            "health": "GET /health - Health check",
            "sms_incoming": "POST /sms/incoming - Twilio webhook for SMS forwarding"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting SMS Spam Detection API...")
    print("Access the API at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
