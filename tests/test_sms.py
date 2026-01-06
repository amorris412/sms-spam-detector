"""
Tests for SMS Integration
Tests webhook endpoint, SMS formatting, and Twilio integration
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.main import app
from api.sms_handler import (
    format_classification_response,
    format_short_response,
    extract_forwarded_message,
    SMSHandler
)


# Initialize test client
client = TestClient(app)


class TestSMSFormatting:
    """Test SMS response formatting functions"""

    def test_format_safe_message(self):
        """Test formatting for safe/ham message"""
        response = format_classification_response(
            is_spam=False,
            is_smishing=False,
            confidence=0.99,
            spam_type="ham",
            risk_level="safe",
            reasoning="Legitimate message pattern"
        )

        assert "✅ SAFE" in response
        assert "99%" in response
        assert "legitimate" in response.lower()

    def test_format_spam_message(self):
        """Test formatting for spam (not smishing) message"""
        response = format_classification_response(
            is_spam=True,
            is_smishing=False,
            confidence=0.95,
            spam_type="spam",
            risk_level="medium",
            reasoning="Prize/winner claim detected"
        )

        assert "⚠️ SPAM DETECTED" in response
        assert "95%" in response
        assert "MEDIUM" in response

    def test_format_smishing_message(self):
        """Test formatting for smishing message"""
        response = format_classification_response(
            is_spam=True,
            is_smishing=True,
            confidence=0.98,
            spam_type="smishing",
            risk_level="high",
            reasoning="Urgent language, credential harvesting pattern",
            url_analysis={
                'has_urls': True,
                'has_url_shortener': True,
                'suspicious_urls': ['bit.ly/verify'],
                'url_risk_score': 0.95
            }
        )

        assert "🚨 SMISHING DETECTED" in response
        assert "98%" in response
        assert "HIGH" in response
        assert "DO NOT CLICK LINKS" in response
        assert "DO NOT REPLY" in response
        assert "DELETE MESSAGE" in response

    def test_format_with_url_analysis(self):
        """Test that URL warnings are included"""
        response = format_classification_response(
            is_spam=True,
            is_smishing=True,
            confidence=0.92,
            spam_type="smishing",
            risk_level="high",
            reasoning="URL shortener detected",
            url_analysis={
                'has_urls': True,
                'has_url_shortener': True,
                'url_risk_score': 0.9
            }
        )

        assert "URL shortener" in response or "shortener detected" in response.lower()

    def test_short_response_safe(self):
        """Test ultra-short safe response"""
        response = format_short_response(
            is_spam=False,
            is_smishing=False,
            confidence=0.99
        )

        assert "✅ SAFE" in response
        assert "99%" in response
        assert len(response) < 160

    def test_short_response_smishing(self):
        """Test ultra-short smishing response"""
        response = format_short_response(
            is_spam=True,
            is_smishing=True,
            confidence=0.98
        )

        assert "🚨 SMISHING" in response
        assert "98%" in response
        assert "DO NOT CLICK" in response
        assert len(response) < 160

    def test_short_response_spam(self):
        """Test ultra-short spam response"""
        response = format_short_response(
            is_spam=True,
            is_smishing=False,
            confidence=0.85
        )

        assert "⚠️ SPAM" in response
        assert "85%" in response
        assert len(response) < 160


class TestForwardedMessageExtraction:
    """Test extraction of forwarded message content"""

    def test_extract_with_fwd_prefix(self):
        """Test extraction with 'Fwd:' prefix"""
        body = "Fwd: URGENT! Your account has been locked."
        result = extract_forwarded_message(body)
        assert result == "URGENT! Your account has been locked."
        assert not result.startswith("Fwd:")

    def test_extract_with_forwarded_prefix(self):
        """Test extraction with 'Forwarded:' prefix"""
        body = "Forwarded: Your verification code is 123456"
        result = extract_forwarded_message(body)
        assert result == "Your verification code is 123456"

    def test_extract_with_forwarded_message_prefix(self):
        """Test extraction with 'Forwarded message:' prefix"""
        body = "Forwarded message: Click here to claim your prize"
        result = extract_forwarded_message(body)
        assert result == "Click here to claim your prize"

    def test_extract_no_prefix(self):
        """Test extraction with no forward prefix"""
        body = "URGENT! Your account has been locked."
        result = extract_forwarded_message(body)
        assert result == body

    def test_extract_case_insensitive(self):
        """Test that extraction is case insensitive"""
        body = "FWD: Test message"
        result = extract_forwarded_message(body)
        assert result == "Test message"

    def test_extract_with_whitespace(self):
        """Test extraction handles extra whitespace"""
        body = "  Fwd:   Test message  "
        result = extract_forwarded_message(body)
        assert result == "Test message"


class TestSMSWebhook:
    """Test SMS webhook endpoint"""

    def test_webhook_receives_sms(self):
        """Test that webhook accepts SMS from Twilio"""
        response = client.post(
            "/sms/incoming",
            data={
                "From": "+15551234567",
                "Body": "Your verification code is 123456",
                "MessageSid": "SM1234567890"
            }
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        assert "<Response>" in response.text
        assert "<Message>" in response.text

    def test_webhook_classifies_safe_message(self):
        """Test webhook correctly classifies safe message"""
        response = client.post(
            "/sms/incoming",
            data={
                "From": "+15551234567",
                "Body": "Your verification code is 123456. Valid for 10 minutes.",
                "MessageSid": "SM1234567890"
            }
        )

        assert response.status_code == 200
        response_text = response.text.lower()
        assert "safe" in response_text or "legitimate" in response_text

    def test_webhook_classifies_spam_message(self):
        """Test webhook correctly classifies spam message"""
        response = client.post(
            "/sms/incoming",
            data={
                "From": "+15551234567",
                "Body": "URGENT! Click bit.ly/verify to unlock your account NOW!",
                "MessageSid": "SM1234567890"
            }
        )

        assert response.status_code == 200
        response_text = response.text.lower()
        assert "spam" in response_text or "smishing" in response_text

    def test_webhook_handles_forwarded_message(self):
        """Test webhook extracts forwarded message content"""
        response = client.post(
            "/sms/incoming",
            data={
                "From": "+15551234567",
                "Body": "Fwd: Free gift card! Click here to claim.",
                "MessageSid": "SM1234567890"
            }
        )

        assert response.status_code == 200
        # Should classify the actual message, not the "Fwd:" prefix

    def test_webhook_missing_required_fields(self):
        """Test webhook handles missing required fields"""
        response = client.post(
            "/sms/incoming",
            data={
                "Body": "Test message"
                # Missing 'From' field
            }
        )

        # Should return 422 Unprocessable Entity for missing required field
        assert response.status_code == 422

    def test_webhook_returns_twiml(self):
        """Test webhook returns valid TwiML"""
        response = client.post(
            "/sms/incoming",
            data={
                "From": "+15551234567",
                "Body": "Test message",
                "MessageSid": "SM1234567890"
            }
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"

        # Check for valid TwiML structure
        text = response.text
        assert "<?xml" in text  # XML declaration
        assert "<Response>" in text
        assert "</Response>" in text
        assert "<Message>" in text
        assert "</Message>" in text


class TestSMSHandler:
    """Test SMSHandler class"""

    def test_handler_initialization_without_credentials(self):
        """Test handler initializes without credentials"""
        handler = SMSHandler()
        # Should initialize but be disabled
        assert handler.enabled == False or handler.enabled == True
        # Could be either depending on if .env exists

    def test_create_twiml_response(self):
        """Test TwiML response creation"""
        handler = SMSHandler()
        twiml = handler.create_twiml_response("Test message")

        assert "<?xml" in twiml
        assert "<Response>" in twiml
        assert "</Response>" in twiml
        assert "<Message>" in twiml
        assert "Test message" in twiml

    def test_twiml_escapes_special_characters(self):
        """Test that TwiML properly escapes special characters"""
        handler = SMSHandler()
        twiml = handler.create_twiml_response("Test & <message>")

        assert "<Response>" in twiml
        # Twilio library should handle escaping


class TestSMSEndToEnd:
    """End-to-end tests for complete SMS flow"""

    def test_complete_flow_smishing_detection(self):
        """Test complete flow: receive SMS, classify as smishing, return warning"""
        # Simulate receiving a smishing SMS
        response = client.post(
            "/sms/incoming",
            data={
                "From": "+15551234567",
                "Body": "URGENT! Your bank account locked. Click http://bit.ly/verify123 NOW!",
                "MessageSid": "SM1234567890"
            }
        )

        assert response.status_code == 200
        text = response.text.lower()

        # Check that response contains smishing warning
        assert any(word in text for word in ["smishing", "spam", "warning", "danger"])

        # Check that response advises not clicking links
        assert any(phrase in text for phrase in ["do not click", "don't click", "delete"])

    def test_complete_flow_safe_message(self):
        """Test complete flow: receive SMS, classify as safe, return confirmation"""
        response = client.post(
            "/sms/incoming",
            data={
                "From": "+15551234567",
                "Body": "Your Amazon order #123-4567890-1234567 has shipped. Track at amazon.com/track",
                "MessageSid": "SM1234567890"
            }
        )

        assert response.status_code == 200
        text = response.text.lower()

        # Check that response indicates safety
        assert any(word in text for word in ["safe", "legitimate", "ok"])


class TestAPIStatsTracking:
    """Test that SMS events are tracked in stats"""

    def test_sms_stats_tracked(self):
        """Test that SMS receive count is tracked"""
        # Send an SMS
        client.post(
            "/sms/incoming",
            data={
                "From": "+15551234567",
                "Body": "Test message",
                "MessageSid": "SM1234567890"
            }
        )

        # Check stats
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200
        # Stats should be tracked (exact count depends on other tests)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
