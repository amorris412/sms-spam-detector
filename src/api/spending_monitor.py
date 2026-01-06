"""
Twilio Spending Monitor
Tracks daily SMS spending and disables service if limit exceeded
"""

import os
from datetime import datetime, date
from pathlib import Path
import json
from typing import Optional
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()


class SpendingMonitor:
    """Monitor Twilio spending and enforce daily limits"""

    def __init__(self, daily_limit: float = 5.0, alert_phone: str = None):
        """
        Initialize spending monitor

        Args:
            daily_limit: Maximum daily spend in USD (default: $5)
            alert_phone: Phone number to send alerts to (default: Twilio number)
        """
        self.daily_limit = daily_limit
        self.alert_phone = alert_phone or os.getenv('TWILIO_PHONE_NUMBER')

        # Twilio client
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.phone_number = os.getenv('TWILIO_PHONE_NUMBER')

        if account_sid and auth_token:
            self.client = Client(account_sid, auth_token)
            self.enabled = True
        else:
            self.client = None
            self.enabled = False

        # State file to track spending
        self.state_file = Path("data/spending_state.json")
        self.state_file.parent.mkdir(exist_ok=True)

        # Load or initialize state
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load spending state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {
            'date': str(date.today()),
            'daily_spend': 0.0,
            'message_count': 0,
            'disabled': False,
            'alert_sent': False
        }

    def _save_state(self):
        """Save spending state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _reset_if_new_day(self):
        """Reset counters if it's a new day"""
        today = str(date.today())
        if self.state['date'] != today:
            self.state = {
                'date': today,
                'daily_spend': 0.0,
                'message_count': 0,
                'disabled': False,
                'alert_sent': False
            }
            self._save_state()
            print(f"📅 New day: Spending monitor reset for {today}")

    def is_service_enabled(self) -> bool:
        """Check if SMS service is currently enabled"""
        self._reset_if_new_day()
        return not self.state['disabled']

    def record_message(self, cost: float = 0.015):
        """
        Record an SMS message and update spending

        Args:
            cost: Cost per message (default: $0.015 = $0.0075 receive + $0.0075 send)
        """
        if not self.enabled:
            return

        self._reset_if_new_day()

        self.state['daily_spend'] += cost
        self.state['message_count'] += 1
        self._save_state()

        print(f"💰 Daily spend: ${self.state['daily_spend']:.4f} ({self.state['message_count']} messages)")

        # Check if limit exceeded
        if self.state['daily_spend'] >= self.daily_limit and not self.state['alert_sent']:
            self._handle_limit_exceeded()

    def _handle_limit_exceeded(self):
        """Handle case when spending limit is exceeded"""
        print(f"\n⚠️  SPENDING LIMIT EXCEEDED!")
        print(f"   Daily limit: ${self.daily_limit:.2f}")
        print(f"   Current spend: ${self.state['daily_spend']:.2f}")
        print(f"   Messages today: {self.state['message_count']}")

        # Disable service
        self.state['disabled'] = True
        self.state['alert_sent'] = True
        self._save_state()

        # Send alert SMS
        self._send_alert()

        print(f"   🚫 SMS service disabled until midnight")

    def _send_alert(self):
        """Send alert SMS about spending limit"""
        if not self.enabled:
            return

        try:
            message = (
                f"ALERT: Daily Twilio spending limit exceeded!\n\n"
                f"Limit: ${self.daily_limit:.2f}\n"
                f"Spent: ${self.state['daily_spend']:.2f}\n"
                f"Messages: {self.state['message_count']}\n\n"
                f"SMS spam detection service has been disabled until midnight to prevent further charges."
            )

            self.client.messages.create(
                body=message,
                from_=self.phone_number,
                to=self.alert_phone
            )

            print(f"   ✅ Alert sent to {self.alert_phone}")

        except Exception as e:
            print(f"   ❌ Failed to send alert: {e}")

    def get_status(self) -> dict:
        """Get current spending status"""
        self._reset_if_new_day()

        return {
            'enabled': not self.state['disabled'],
            'daily_limit': self.daily_limit,
            'daily_spend': self.state['daily_spend'],
            'message_count': self.state['message_count'],
            'remaining_budget': max(0, self.daily_limit - self.state['daily_spend']),
            'date': self.state['date']
        }

    def manually_enable(self):
        """Manually re-enable service (admin override)"""
        self.state['disabled'] = False
        self._save_state()
        print("✅ SMS service manually re-enabled")

    def manually_disable(self):
        """Manually disable service (admin override)"""
        self.state['disabled'] = True
        self._save_state()
        print("🚫 SMS service manually disabled")

    def get_twilio_usage(self) -> Optional[dict]:
        """
        Get actual usage from Twilio API

        Returns:
            Dict with usage stats or None if unavailable
        """
        if not self.enabled:
            return None

        try:
            today = date.today()

            # Get messages from today
            messages = self.client.messages.list(
                date_sent_after=datetime(today.year, today.month, today.day)
            )

            total_cost = 0.0
            sms_count = len(messages)

            for msg in messages:
                if msg.price:
                    # Price is returned as string like "-0.0075"
                    total_cost += abs(float(msg.price))

            return {
                'message_count': sms_count,
                'total_cost': total_cost,
                'date': str(today)
            }

        except Exception as e:
            print(f"Warning: Could not fetch Twilio usage: {e}")
            return None
