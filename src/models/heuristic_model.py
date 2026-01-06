"""
Heuristic Spam/Smishing Detection Model
Combines text pattern matching with URL analysis

Features:
- Text-based heuristics (urgent language, spam keywords, etc.)
- URL analysis (12 URL features)
- OTP validation (ensures 2FA messages aren't flagged)
- Fast inference (<5ms typical)
- Interpretable results
"""

import re
import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from features.url_features import URLFeatureExtractor


class HeuristicSpamDetector:
    """Rule-based spam detector with URL analysis"""
    
    # Spam keywords (high weight)
    SPAM_KEYWORDS = {
        'free', 'win', 'winner', 'prize', 'congratulations', 'claim',
        'cash', 'money', 'earn', '$$$', 'discount', 'offer',
        'click here', 'click now', 'act now', 'limited time',
        'call now', 'txt', 'text back', 'reply', 'opt in'
    }
    
    # Smishing keywords (very high weight)
    SMISHING_KEYWORDS = {
        'verify', 'confirm', 'suspended', 'locked', 'blocked',
        'unusual activity', 'unauthorized', 'security alert',
        'account update', 'payment failed', 'expired', 'update payment',
        'urgent', 'immediate', 'action required', 'within 24',
        'restore access', 'reactivate', 'suspended account'
    }
    
    # Legitimate A2P patterns (negative weight - reduces spam score)
    LEGITIMATE_PATTERNS = {
        'verification code', 'otp', 'one-time password',
        'authentication code', 'security code', 'confirmation code',
        'order confirmation', 'booking confirmed', 'appointment reminder',
        'delivery scheduled', 'shipped', 'tracking number',
        'payment received', 'refund processed', 'subscription renewed',
        'reply stop to unsubscribe', 'text stop to opt out'
    }
    
    # OTP pattern - CRITICAL: Do not flag these as spam
    OTP_PATTERN = re.compile(r'\b\d{4,8}\b')
    
    def __init__(self):
        self.url_extractor = URLFeatureExtractor()
    
    def extract_text_features(self, message: str) -> Dict[str, float]:
        """Extract text-based heuristic features"""
        message_lower = message.lower()
        
        features = {}
        
        # Feature 1: Message length (very short or very long is suspicious)
        length = len(message)
        if length < 20:
            features['length_score'] = 0.3
        elif length > 200:
            features['length_score'] = 0.4
        else:
            features['length_score'] = 0.0
        
        # Feature 2: ALL CAPS ratio
        if length > 0:
            caps_ratio = sum(1 for c in message if c.isupper()) / length
            features['caps_ratio'] = min(caps_ratio * 2, 1.0)  # Normalize
        else:
            features['caps_ratio'] = 0.0
        
        # Feature 3: Exclamation marks
        exclamation_count = message.count('!')
        features['exclamation_score'] = min(exclamation_count / 3.0, 1.0)
        
        # Feature 4: Spam keywords
        spam_keyword_count = sum(
            1 for keyword in self.SPAM_KEYWORDS
            if keyword in message_lower
        )
        features['spam_keywords'] = min(spam_keyword_count / 3.0, 1.0)
        
        # Feature 5: Smishing keywords (higher weight)
        smishing_keyword_count = sum(
            1 for keyword in self.SMISHING_KEYWORDS
            if keyword in message_lower
        )
        features['smishing_keywords'] = min(smishing_keyword_count / 2.0, 1.0)
        
        # Feature 6: Legitimate patterns (negative - reduces spam score)
        legitimate_count = sum(
            1 for pattern in self.LEGITIMATE_PATTERNS
            if pattern in message_lower
        )
        features['legitimate_patterns'] = min(legitimate_count / 2.0, 1.0)
        
        # Feature 7: Numbers (often in spam for prizes/amounts)
        number_count = len(re.findall(r'\d+', message))
        features['number_score'] = min(number_count / 5.0, 1.0)
        
        # Feature 8: Special characters
        special_count = sum(1 for c in message if c in '$£€@#%&*')
        features['special_chars'] = min(special_count / 5.0, 1.0)
        
        # Feature 9: Urgency indicators
        urgency_words = ['urgent', 'now', 'immediately', 'asap', 'hurry', 'fast', 'quick']
        urgency_count = sum(1 for word in urgency_words if word in message_lower)
        features['urgency_score'] = min(urgency_count / 2.0, 1.0)
        
        # Feature 10: Money mentions
        money_patterns = ['$', '£', '€', 'dollars', 'cash', 'money', 'prize', 'win']
        money_count = sum(1 for pattern in money_patterns if pattern in message_lower)
        features['money_score'] = min(money_count / 2.0, 1.0)
        
        return features
    
    def is_otp_message(self, message: str) -> bool:
        """
        Check if message is a legitimate OTP/2FA message
        CRITICAL: These should NEVER be flagged as spam/smishing
        """
        message_lower = message.lower()
        
        # Must have OTP-related keywords
        otp_indicators = [
            'verification code', 'authentication code', 'security code',
            'otp', 'one-time password', 'one time password',
            'confirm', 'verify your'
        ]
        
        has_otp_keyword = any(indicator in message_lower for indicator in otp_indicators)
        
        # Must have a numeric code
        has_numeric_code = bool(self.OTP_PATTERN.search(message))
        
        # Should NOT have suspicious URLs or other smishing indicators
        urls = self.url_extractor.extract_urls(message)
        has_suspicious_url = False
        if urls:
            url_analysis = self.url_extractor.get_url_analysis(message)
            has_suspicious_url = url_analysis['risk_level'] in ['high', 'medium']
        
        # Should NOT have urgent language beyond verification
        smishing_indicators = [
            'suspended', 'locked', 'unusual activity', 'unauthorized',
            'account update', 'payment failed', 'click here', 'click now'
        ]
        has_smishing_language = any(ind in message_lower for ind in smishing_indicators)
        
        # Valid OTP: has OTP keyword + numeric code, NO suspicious URLs/language
        is_valid_otp = (
            has_otp_keyword and 
            has_numeric_code and 
            not has_suspicious_url and
            not has_smishing_language
        )
        
        return is_valid_otp
    
    def predict(self, message: str) -> Tuple[str, float, Dict]:
        """
        Predict if message is spam/smishing/ham
        
        Returns:
            label: 'spam', 'smishing', or 'ham'
            confidence: 0.0 to 1.0
            details: Dictionary with feature breakdown and reasoning
        """
        # CRITICAL: Check for OTP first
        if self.is_otp_message(message):
            return 'ham', 0.99, {
                'reasoning': 'Legitimate OTP/2FA message',
                'otp_validated': True,
                'risk_level': 'safe'
            }
        
        # Extract all features
        text_features = self.extract_text_features(message)
        url_features = self.url_extractor.extract_message_url_features(message)
        
        # Calculate weighted spam score (optimized weights)
        weights = {
            # Text features (adjusted for better recall)
            'length_score': 0.2,  # Reduced from 0.3
            'caps_ratio': 0.6,  # Increased from 0.5
            'exclamation_score': 0.5,  # Increased from 0.4
            'spam_keywords': 0.9,  # Increased from 0.7
            'smishing_keywords': 1.5,  # Increased from 1.2
            'legitimate_patterns': -2.0,  # Increased negative weight from -1.5
            'number_score': 0.2,  # Reduced from 0.3
            'special_chars': 0.5,  # Increased from 0.4
            'urgency_score': 1.0,  # Increased from 0.8
            'money_score': 0.8,  # Increased from 0.6
            
            # URL features (slightly adjusted)
            'has_urls': 0.3,  # Increased from 0.2
            'num_urls': 0.5,  # Increased from 0.4
            'max_url_risk': 1.8,  # Increased from 1.5
            'has_shortener': 1.0,  # Increased from 0.8
            'has_suspicious_tld': 0.8,  # Increased from 0.7
            'has_ip_url': 1.0,  # Increased from 0.9
            'has_legitimate_domain': -1.0,  # Increased negative from -0.8
            'max_domain_entropy': 0.7,  # Increased from 0.6
            'has_suspicious_url_keywords': 0.8,  # Increased from 0.7
        }
        
        # Combine all features
        all_features = {**text_features, **url_features}
        
        # Calculate spam score
        spam_score = 0.0
        for feature, value in all_features.items():
            if feature in weights:
                spam_score += value * weights[feature]
        
        # Normalize to 0-1 range (adjusted normalization)
        spam_score = max(0, min(spam_score / 10.0, 1.0))  # Changed from 8.0 to 10.0
        
        # Determine label and confidence (optimized thresholds)
        if spam_score < 0.25:  # Lowered from 0.3
            label = 'ham'
            confidence = 1.0 - spam_score
        elif spam_score > 0.55:  # Lowered from 0.7 for better recall
            # Distinguish between spam and smishing
            smishing_indicators = (
                text_features.get('smishing_keywords', 0) > 0.2 or  # Lowered from 0.3
                url_features.get('max_url_risk', 0) > 0.5 or  # Lowered from 0.6
                url_features.get('has_shortener', 0) > 0
            )
            
            if smishing_indicators:
                label = 'smishing'
            else:
                label = 'spam'
            
            confidence = spam_score
        else:
            # Uncertain - now favor spam detection (changed from ham)
            if spam_score > 0.4:  # New threshold
                label = 'spam'
            else:
                label = 'ham'
            confidence = spam_score if spam_score > 0.4 else 1.0 - spam_score
        
        # Build reasoning
        reasoning_parts = []
        
        # Text indicators
        if text_features.get('smishing_keywords', 0) > 0.3:
            reasoning_parts.append('Smishing language detected')
        if text_features.get('spam_keywords', 0) > 0.3:
            reasoning_parts.append('Spam keywords present')
        if text_features.get('urgency_score', 0) > 0.5:
            reasoning_parts.append('Urgent/pressure tactics')
        if text_features.get('legitimate_patterns', 0) > 0.5:
            reasoning_parts.append('Legitimate business patterns')
        
        # URL indicators
        if url_features.get('has_urls', 0) and url_features.get('max_url_risk', 0) > 0.5:
            reasoning_parts.append('Suspicious URL detected')
        if url_features.get('has_shortener', 0):
            reasoning_parts.append('URL shortener used')
        if url_features.get('has_legitimate_domain', 0):
            reasoning_parts.append('Legitimate domain present')
        
        if not reasoning_parts:
            reasoning_parts.append('No significant spam indicators')
        
        # Determine risk level
        if spam_score > 0.7:
            risk_level = 'high'
        elif spam_score > 0.4:
            risk_level = 'medium'
        elif spam_score > 0.2:
            risk_level = 'low'
        else:
            risk_level = 'safe'
        
        details = {
            'spam_score': float(spam_score),
            'risk_level': risk_level,
            'reasoning': ', '.join(reasoning_parts),
            'text_features': {k: float(v) for k, v in text_features.items()},
            'url_features': {k: float(v) for k, v in url_features.items()},
            'url_analysis': self.url_extractor.get_url_analysis(message),
            'otp_validated': False
        }
        
        return label, float(confidence), details
    
    def predict_batch(self, messages):
        """Predict for a batch of messages"""
        results = []
        for message in messages:
            label, confidence, details = self.predict(message)
            results.append({
                'message': message,
                'label': label,
                'confidence': confidence,
                'details': details
            })
        return results


if __name__ == "__main__":
    # Test the heuristic model
    detector = HeuristicSpamDetector()
    
    test_messages = [
        # Smishing
        "URGENT! Your account has been locked. Click bit.ly/verify to restore access NOW!",
        "Your Bank of America account shows unusual activity. Verify: secure-bank.xyz/login",
        
        # Spam
        "Congratulations! You've won $5000! Claim your prize now by texting WIN!",
        "SALE! 50% off everything! Limited time offer! Click here now!",
        
        # Legitimate OTP (should be ham)
        "Your verification code is 123456. Valid for 10 minutes.",
        "Your Google authentication code is 987654.",
        
        # Legitimate transactional
        "Your Amazon order #12345 has shipped. Track: https://www.amazon.com/track/12345",
        "Appointment reminder: Dentist tomorrow at 2pm. Reply C to confirm.",
        
        # Legitimate marketing (with opt-out)
        "Target: Weekend sale! 20% off all clothing. Shop now. Reply STOP to unsubscribe.",
        
        # Ham
        "Hey, are we still meeting at 3pm for coffee?",
        "Don't forget to pick up milk on your way home!",
    ]
    
    print("Heuristic Spam Detector Tests")
    print("=" * 80)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}:")
        print(f"Message: {message[:70]}...")
        
        label, confidence, details = detector.predict(message)
        
        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Risk Level: {details['risk_level']}")
        print(f"Reasoning: {details['reasoning']}")
        
        if details.get('otp_validated'):
            print("✅ Validated as legitimate OTP")
        
        if details['url_analysis']['has_urls']:
            print(f"URLs detected: {details['url_analysis']['num_urls']}")
            print(f"URL risk: {details['url_analysis']['risk_level']}")
    
    print("\n" + "=" * 80)
    print("Heuristic model test complete!")
