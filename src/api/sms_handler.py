"""
SMS Handler for Twilio Integration
Handles incoming SMS messages and sends responses via Twilio
"""

from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from typing import Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()


class SMSHandler:
    """Handle SMS operations via Twilio"""

    def __init__(self):
        """Initialize Twilio client"""
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.phone_number = os.getenv('TWILIO_PHONE_NUMBER')

        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
            self.enabled = True
        else:
            self.client = None
            self.enabled = False
            print("⚠️  Twilio credentials not found. SMS features disabled.")

    def send_sms(self, to_number: str, message: str) -> bool:
        """
        Send SMS via Twilio

        Args:
            to_number: Recipient phone number (E.164 format)
            message: Message text (max 1600 chars for multipart)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            print(f"SMS would be sent to {to_number}: {message}")
            return False

        try:
            message_obj = self.client.messages.create(
                body=message,
                from_=self.phone_number,
                to=to_number
            )
            print(f"✅ SMS sent to {to_number}. SID: {message_obj.sid}")
            return True

        except Exception as e:
            print(f"❌ Failed to send SMS to {to_number}: {e}")
            return False

    def create_twiml_response(self, message: str) -> str:
        """
        Create TwiML response for webhook
        Twilio will automatically send this message back to the sender

        Args:
            message: Response message text

        Returns:
            TwiML XML string
        """
        resp = MessagingResponse()
        resp.message(message)
        return str(resp)


def format_classification_response(
    is_spam: bool,
    is_smishing: bool,
    confidence: float,
    spam_type: str,
    risk_level: str,
    reasoning: str,
    url_analysis: Dict = None
) -> str:
    """
    Format classification result for SMS response
    Optimized for readability in SMS format

    Args:
        Classification results from the model

    Returns:
        Formatted SMS message (may be multipart if >160 chars)
    """

    # Build response based on classification
    if not is_spam:
        # Safe message
        response = f"✅ SAFE ({confidence*100:.0f}% confidence)\n\n"
        response += "This message appears legitimate."
        return response

    # Spam or smishing detected
    if is_smishing:
        icon = "🚨"
        label = "SMISHING DETECTED"
    else:
        icon = "⚠️"
        label = "SPAM DETECTED"

    response = f"{icon} {label}\n"
    response += f"Confidence: {confidence*100:.0f}%\n"
    response += f"Risk: {risk_level.upper()}\n\n"

    # Add key warning indicators (limit to most important)
    warnings = []

    if url_analysis and url_analysis.get('has_urls'):
        if url_analysis.get('has_url_shortener'):
            warnings.append("• URL shortener detected")
        if url_analysis.get('suspicious_urls'):
            warnings.append(f"• Suspicious link found")
        if url_analysis.get('url_risk_score', 0) > 0.7:
            warnings.append("• High-risk URL")

    # Parse reasoning for key indicators
    reasoning_lower = reasoning.lower()
    if 'urgent' in reasoning_lower or 'immediate' in reasoning_lower:
        warnings.append("• Urgent language")
    if 'prize' in reasoning_lower or 'winner' in reasoning_lower:
        warnings.append("• Prize/winner claim")
    if 'account' in reasoning_lower and 'verify' in reasoning_lower:
        warnings.append("• Account verification request")
    if 'credential' in reasoning_lower:
        warnings.append("• Credential harvesting")

    # Limit to top 4 warnings
    if warnings:
        response += "\n".join(warnings[:4]) + "\n"

    # Add action advice
    if is_smishing:
        response += "\n⚠️ DO NOT CLICK LINKS"
        response += "\n⚠️ DO NOT REPLY"
        response += "\n⚠️ DELETE MESSAGE"
    else:
        response += "\nRecommend: Delete message"

    return response


def format_short_response(is_spam: bool, is_smishing: bool, confidence: float) -> str:
    """
    Ultra-short response for when brevity is needed

    Returns:
        Single SMS message (<160 chars)
    """
    if not is_spam:
        return f"✅ SAFE ({confidence*100:.0f}%)"

    if is_smishing:
        return f"🚨 SMISHING! ({confidence*100:.0f}%) - DO NOT CLICK LINKS OR REPLY"
    else:
        return f"⚠️ SPAM ({confidence*100:.0f}%) - Recommend deleting"


def extract_forwarded_message(body: str) -> str:
    """
    Extract the actual message content from a forwarded SMS
    Some phones add "Fwd:" or "Forwarded message:" prefixes

    Args:
        body: Raw SMS body received via webhook

    Returns:
        Cleaned message content
    """
    body = body.strip()

    # Remove common forward prefixes
    prefixes = [
        'fwd:',
        'forwarded:',
        'forwarded message:',
        'fwd message:',
        '> ',  # Some phones use this
    ]

    body_lower = body.lower()
    for prefix in prefixes:
        if body_lower.startswith(prefix):
            body = body[len(prefix):].strip()
            break

    return body
