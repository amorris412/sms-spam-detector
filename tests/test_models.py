"""
Tests for SMS Spam Detection Models
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.heuristic_model import HeuristicSpamDetector
from features.url_features import URLFeatureExtractor


class TestHeuristicModel:
    """Test heuristic spam detector"""
    
    @pytest.fixture
    def detector(self):
        return HeuristicSpamDetector()
    
    def test_smishing_detection(self, detector):
        """Test that obvious smishing is detected"""
        message = "URGENT! Your account locked. Click bit.ly/verify NOW!"
        label, confidence, details = detector.predict(message)
        
        assert label in ['spam', 'smishing'], f"Expected spam/smishing, got {label}"
        assert confidence > 0.7, f"Low confidence: {confidence}"
        assert details['risk_level'] in ['high', 'medium']
    
    def test_legitimate_otp(self, detector):
        """Test that legitimate OTP is not flagged"""
        message = "Your verification code is 123456. Valid for 10 minutes."
        label, confidence, details = detector.predict(message)
        
        assert label == 'ham', f"OTP incorrectly classified as {label}"
        assert details.get('otp_validated', False), "OTP validation failed"
    
    def test_legitimate_transactional(self, detector):
        """Test legitimate transactional messages"""
        message = "Your Amazon order #12345 has shipped. Track at https://www.amazon.com/track"
        label, confidence, details = detector.predict(message)
        
        assert label == 'ham', f"Transactional message classified as {label}"
    
    def test_spam_keywords(self, detector):
        """Test spam keyword detection"""
        message = "Congratulations! You've won $5000! Text WIN now!"
        label, confidence, details = detector.predict(message)
        
        assert label in ['spam', 'smishing']
        assert confidence > 0.5
    
    def test_url_shortener_detection(self, detector):
        """Test that URL shorteners are flagged"""
        message = "Check this out: bit.ly/suspicious123"
        label, confidence, details = detector.predict(message)
        
        url_analysis = details.get('url_analysis', {})
        assert url_analysis.get('has_urls'), "URL not detected"
        assert label in ['spam', 'smishing'] or url_analysis.get('risk_level') != 'safe'


class TestURLFeatureExtractor:
    """Test URL feature extraction"""
    
    @pytest.fixture
    def extractor(self):
        return URLFeatureExtractor()
    
    def test_url_extraction(self, extractor):
        """Test basic URL extraction"""
        message = "Visit https://example.com and http://test.com"
        urls = extractor.extract_urls(message)
        
        assert len(urls) == 2
        assert any('example.com' in url for url in urls)
        assert any('test.com' in url for url in urls)
    
    def test_shortener_detection(self, extractor):
        """Test URL shortener detection"""
        url = "https://bit.ly/test123"
        features = extractor.extract_url_features(url)
        
        assert features['is_shortener'] == 1.0, "Shortener not detected"
    
    def test_suspicious_tld(self, extractor):
        """Test suspicious TLD detection"""
        url = "https://suspicious.xyz/verify"
        features = extractor.extract_url_features(url)
        
        assert features['suspicious_tld'] == 1.0, "Suspicious TLD not detected"
    
    def test_legitimate_domain(self, extractor):
        """Test legitimate domain recognition"""
        url = "https://www.amazon.com/product"
        features = extractor.extract_url_features(url)
        
        assert features['is_legitimate_domain'] == 1.0, "Legitimate domain not recognized"
        assert features['url_risk_score'] < 0.3, "Risk score too high for legitimate domain"
    
    def test_ip_address_detection(self, extractor):
        """Test IP address in URL detection"""
        url = "http://192.168.1.1/phishing"
        features = extractor.extract_url_features(url)
        
        assert features['is_ip_address'] == 1.0, "IP address not detected"
    
    def test_https_detection(self, extractor):
        """Test HTTPS detection"""
        url_https = "https://example.com"
        url_http = "http://example.com"
        
        features_https = extractor.extract_url_features(url_https)
        features_http = extractor.extract_url_features(url_http)
        
        assert features_https['has_https'] == 1.0
        assert features_http['has_https'] == 0.0


class TestModelIntegration:
    """Integration tests"""
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        detector = HeuristicSpamDetector()
        
        messages = [
            "URGENT! Click now!",
            "Hey, are we meeting at 3pm?",
            "Your code is 123456"
        ]
        
        results = detector.predict_batch(messages)
        
        assert len(results) == len(messages)
        assert all('label' in r for r in results)
        assert all('confidence' in r for r in results)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
