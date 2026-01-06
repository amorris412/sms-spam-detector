"""
Message Tracker
Tracks detailed message statistics for dashboard
"""

import json
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional


class MessageTracker:
    """Track message statistics for dashboard"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / "message_history.jsonl"
        self.daily_stats_file = self.data_dir / "daily_stats.json"

        # In-memory stats
        self.stats = {
            'total_messages': 0,
            'spam_count': 0,
            'smishing_count': 0,
            'ham_count': 0,
            'sms_received': 0,
            'date': str(date.today())
        }

        # Load existing stats
        self._load_daily_stats()

    def _load_daily_stats(self):
        """Load daily statistics from file"""
        if self.daily_stats_file.exists():
            try:
                with open(self.daily_stats_file, 'r') as f:
                    saved_stats = json.load(f)

                # Reset if new day
                if saved_stats.get('date') == str(date.today()):
                    self.stats = saved_stats
                else:
                    # Archive old stats and reset
                    self._archive_daily_stats(saved_stats)
                    self._reset_stats()
            except:
                pass

    def _save_daily_stats(self):
        """Save daily statistics to file"""
        with open(self.daily_stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def _reset_stats(self):
        """Reset statistics for new day"""
        self.stats = {
            'total_messages': 0,
            'spam_count': 0,
            'smishing_count': 0,
            'ham_count': 0,
            'sms_received': 0,
            'date': str(date.today())
        }
        self._save_daily_stats()

    def _archive_daily_stats(self, stats: dict):
        """Archive previous day's stats"""
        archive_file = self.data_dir / "stats_archive.jsonl"
        with open(archive_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')

    def record_message(
        self,
        message: str,
        classification: str,
        is_smishing: bool,
        confidence: float,
        from_number: Optional[str] = None,
        via_sms: bool = False
    ):
        """
        Record a message classification

        Args:
            message: The message text (truncated for privacy)
            classification: 'spam', 'ham', or 'smishing'
            is_smishing: Whether it's smishing
            confidence: Classification confidence
            from_number: Phone number (optional, anonymized)
            via_sms: Whether received via SMS forwarding
        """
        # Check if new day
        if self.stats['date'] != str(date.today()):
            self._archive_daily_stats(self.stats)
            self._reset_stats()

        # Update counts
        self.stats['total_messages'] += 1

        if classification in ['spam', 'smishing'] or is_smishing:
            self.stats['spam_count'] += 1
            if is_smishing:
                self.stats['smishing_count'] += 1
        else:
            self.stats['ham_count'] += 1

        if via_sms:
            self.stats['sms_received'] += 1

        # Save detailed record to history
        record = {
            'timestamp': datetime.now().isoformat(),
            'date': str(date.today()),
            'classification': classification,
            'is_smishing': is_smishing,
            'confidence': confidence,
            'via_sms': via_sms,
            'message_length': len(message),
            'from_number_hash': hash(from_number) if from_number else None
        }

        with open(self.history_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Save updated stats
        self._save_daily_stats()

    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            **self.stats,
            'spam_rate': (
                self.stats['spam_count'] / self.stats['total_messages']
                if self.stats['total_messages'] > 0
                else 0
            ),
            'smishing_rate': (
                self.stats['smishing_count'] / self.stats['spam_count']
                if self.stats['spam_count'] > 0
                else 0
            )
        }

    def get_historical_stats(self, days: int = 7) -> List[Dict]:
        """
        Get historical statistics

        Args:
            days: Number of days to retrieve

        Returns:
            List of daily statistics
        """
        stats = []

        # Get current day
        stats.append(self.stats)

        # Get archived stats
        archive_file = self.data_dir / "stats_archive.jsonl"
        if archive_file.exists():
            try:
                with open(archive_file, 'r') as f:
                    lines = f.readlines()

                # Get last N days
                for line in lines[-days+1:]:
                    stats.append(json.loads(line))

            except:
                pass

        return stats[-days:]

    def get_recent_messages(self, limit: int = 100) -> List[Dict]:
        """
        Get recent message records

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent message records
        """
        records = []

        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    lines = f.readlines()

                # Get last N records
                for line in lines[-limit:]:
                    records.append(json.loads(line))

            except:
                pass

        return records
