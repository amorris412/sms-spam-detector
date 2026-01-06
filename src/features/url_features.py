"""
URL Feature Extraction for Spam/Smishing Detection
Extracts 12 features from URLs to detect malicious patterns

Features:
1. URL shortener detection
2. Suspicious TLD detection
3. IP address in URL
4. Domain entropy (randomness)
5. Suspicious keywords
6. Legitimate domain recognition
7. HTTPS detection
8. Number of subdomains
9. URL length
10. Number of special characters
11. Number of URLs in message
12. Combined URL risk score
"""

import re
from urllib.parse import urlparse
import math
from typing import List, Dict, Tuple


class URLFeatureExtractor:
    """Extract comprehensive URL features for spam/smishing detection"""
    
    # Known URL shorteners
    URL_SHORTENERS = {
        'bit.ly', 'tinyurl.com', 'ow.ly', 'goo.gl', 't.co', 'is.gd',
        'buff.ly', 'adf.ly', 'short.link', 'tiny.cc', 'cli.gs',
        'shorte.st', 'cutt.ly', 'rb.gy', 'shorturl.at'
    }
    
    # Suspicious TLDs (commonly used in phishing)
    SUSPICIOUS_TLDS = {
        '.xyz', '.top', '.club', '.work', '.online', '.site', '.website',
        '.space', '.tech', '.store', '.fun', '.click', '.link', '.gq',
        '.ml', '.cf', '.tk', '.ga', '.bid', '.win', '.download'
    }
    
    # Legitimate domains (whitelist - expanded)
    LEGITIMATE_DOMAINS = {
        # Social media
        'google.com', 'facebook.com', 'instagram.com', 'twitter.com', 
        'linkedin.com', 'youtube.com', 'tiktok.com', 'snapchat.com',
        # E-commerce
        'amazon.com', 'ebay.com', 'walmart.com', 'target.com', 'etsy.com',
        'shopify.com', 'bestbuy.com', 'homedepot.com', 'lowes.com',
        # Tech companies
        'apple.com', 'microsoft.com', 'samsung.com', 'dell.com', 'hp.com',
        # Streaming/Entertainment
        'netflix.com', 'spotify.com', 'hulu.com', 'disney.com', 'hbomax.com',
        # Financial
        'paypal.com', 'venmo.com', 'chase.com', 'wellsfargo.com', 
        'bankofamerica.com', 'citibank.com', 'capitalone.com',
        # Delivery/Logistics  
        'usps.com', 'fedex.com', 'ups.com', 'dhl.com',
        # Food delivery
        'doordash.com', 'grubhub.com', 'ubereats.com', 'instacart.com',
        # Travel
        'uber.com', 'lyft.com', 'airbnb.com', 'booking.com', 'expedia.com',
        # Other major services
        'github.com', 'dropbox.com', 'zoom.us', 'slack.com', 'discord.com'
    }
    
    # Suspicious keywords in URLs (expanded list)
    SUSPICIOUS_KEYWORDS = {
        'verify', 'secure', 'account', 'update', 'confirm', 'login',
        'signin', 'banking', 'password', 'suspended', 'locked',
        'unusual', 'activity', 'urgent', 'expire', 'refund', 'claim',
        'winner', 'prize', 'free', 'offer', 'billing', 'payment',
        'restore', 'reactivate', 'validate', 'authorize', 'alert',
        'warning', 'blocked', 'limited', 'restriction', 'hold',
        'action', 'required', 'immediately', 'click', 'here'
    }
    
    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        return self.url_pattern.findall(text)
    
    def calculate_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string (measures randomness)"""
        if not s:
            return 0.0
        
        # Calculate probability of each character
        prob = {char: s.count(char) / len(s) for char in set(s)}
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in prob.values())
        return entropy
    
    def is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        return bool(ip_pattern.match(domain))
    
    def extract_url_features(self, url: str) -> Dict[str, float]:
        """Extract all features from a single URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            full_url = url.lower()
            
            features = {}
            
            # Feature 1: URL shortener
            features['is_shortener'] = float(any(
                shortener in domain for shortener in self.URL_SHORTENERS
            ))
            
            # Feature 2: Suspicious TLD
            features['suspicious_tld'] = float(any(
                domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS
            ))
            
            # Feature 3: IP address in URL
            features['is_ip_address'] = float(self.is_ip_address(domain))
            
            # Feature 4: Domain entropy (randomness)
            # Remove TLD for entropy calculation
            domain_name = domain.split('.')[0] if '.' in domain else domain
            features['domain_entropy'] = self.calculate_entropy(domain_name)
            # Normalize to 0-1 (entropy typically 0-5 for domains)
            features['domain_entropy'] = min(features['domain_entropy'] / 5.0, 1.0)
            
            # Feature 5: Suspicious keywords
            features['has_suspicious_keywords'] = float(any(
                keyword in full_url for keyword in self.SUSPICIOUS_KEYWORDS
            ))
            
            # Feature 6: Is legitimate domain
            features['is_legitimate_domain'] = float(any(
                legit in domain for legit in self.LEGITIMATE_DOMAINS
            ))
            
            # Feature 7: HTTPS
            features['has_https'] = float(parsed.scheme == 'https')
            
            # Feature 8: Number of subdomains
            parts = domain.split('.')
            # Typical: subdomain.domain.tld (3 parts is normal)
            num_subdomains = max(0, len(parts) - 2)
            features['num_subdomains'] = min(num_subdomains / 3.0, 1.0)  # Normalize
            
            # Feature 9: URL length (long URLs often suspicious)
            features['url_length'] = min(len(url) / 100.0, 1.0)  # Normalize
            
            # Feature 10: Special characters
            special_chars = sum(1 for c in url if c in '@?=&%')
            features['special_chars'] = min(special_chars / 10.0, 1.0)  # Normalize
            
            # Feature 11: Path depth
            path_depth = len([p for p in path.split('/') if p])
            features['path_depth'] = min(path_depth / 5.0, 1.0)  # Normalize
            
            # Feature 12: Combined risk score (weighted)
            risk_weights = {
                'is_shortener': 0.8,
                'suspicious_tld': 0.7,
                'is_ip_address': 0.9,
                'domain_entropy': 0.6,
                'has_suspicious_keywords': 0.7,
                'is_legitimate_domain': -0.9,  # Negative weight (reduces risk)
                'has_https': -0.3,  # Slightly reduces risk
                'num_subdomains': 0.5,
                'url_length': 0.4,
                'special_chars': 0.5,
                'path_depth': 0.3,
            }
            
            risk_score = sum(
                features[k] * risk_weights[k] for k in risk_weights.keys()
            )
            # Normalize to 0-1
            features['url_risk_score'] = max(0, min(risk_score / 5.0, 1.0))
            
            return features
            
        except Exception as e:
            # If URL parsing fails, return high-risk default features
            return {
                'is_shortener': 0.0,
                'suspicious_tld': 0.0,
                'is_ip_address': 0.0,
                'domain_entropy': 0.5,
                'has_suspicious_keywords': 0.0,
                'is_legitimate_domain': 0.0,
                'has_https': 0.0,
                'num_subdomains': 0.0,
                'url_length': 0.5,
                'special_chars': 0.0,
                'path_depth': 0.0,
                'url_risk_score': 0.5
            }
    
    def extract_message_url_features(self, message: str) -> Dict[str, float]:
        """
        Extract URL features from entire message
        Returns aggregated features across all URLs
        """
        urls = self.extract_urls(message)
        
        # Base features
        features = {
            'has_urls': float(len(urls) > 0),
            'num_urls': min(len(urls) / 3.0, 1.0),  # Normalize (3+ URLs is very suspicious)
        }
        
        if not urls:
            # No URLs - set all features to 0
            features.update({
                'max_url_risk': 0.0,
                'avg_url_risk': 0.0,
                'has_shortener': 0.0,
                'has_suspicious_tld': 0.0,
                'has_ip_url': 0.0,
                'has_legitimate_domain': 0.0,
                'all_https': 0.0,
                'max_domain_entropy': 0.0,
                'has_suspicious_url_keywords': 0.0,
            })
        else:
            # Extract features from all URLs
            url_features_list = [self.extract_url_features(url) for url in urls]
            
            # Aggregate features
            features['max_url_risk'] = max(f['url_risk_score'] for f in url_features_list)
            features['avg_url_risk'] = sum(f['url_risk_score'] for f in url_features_list) / len(url_features_list)
            features['has_shortener'] = float(any(f['is_shortener'] for f in url_features_list))
            features['has_suspicious_tld'] = float(any(f['suspicious_tld'] for f in url_features_list))
            features['has_ip_url'] = float(any(f['is_ip_address'] for f in url_features_list))
            features['has_legitimate_domain'] = float(any(f['is_legitimate_domain'] for f in url_features_list))
            features['all_https'] = float(all(f['has_https'] for f in url_features_list))
            features['max_domain_entropy'] = max(f['domain_entropy'] for f in url_features_list)
            features['has_suspicious_url_keywords'] = float(any(f['has_suspicious_keywords'] for f in url_features_list))
        
        return features
    
    def get_url_analysis(self, message: str) -> Dict:
        """
        Get human-readable URL analysis for a message
        Useful for API responses and debugging
        """
        urls = self.extract_urls(message)
        
        if not urls:
            return {
                'has_urls': False,
                'num_urls': 0,
                'urls': [],
                'risk_level': 'none',
                'suspicious_indicators': []
            }
        
        url_details = []
        suspicious_indicators = []
        max_risk = 0.0
        
        for url in urls:
            features = self.extract_url_features(url)
            
            # Collect suspicious indicators
            indicators = []
            if features['is_shortener']:
                indicators.append('URL shortener')
            if features['suspicious_tld']:
                indicators.append('Suspicious domain extension')
            if features['is_ip_address']:
                indicators.append('IP address instead of domain')
            if features['domain_entropy'] > 0.7:
                indicators.append('Random-looking domain')
            if features['has_suspicious_keywords']:
                indicators.append('Suspicious keywords in URL')
            if not features['is_legitimate_domain'] and not features['has_https']:
                indicators.append('No HTTPS encryption')
            
            url_details.append({
                'url': url,
                'risk_score': features['url_risk_score'],
                'indicators': indicators
            })
            
            suspicious_indicators.extend(indicators)
            max_risk = max(max_risk, features['url_risk_score'])
        
        # Determine risk level
        if max_risk > 0.7:
            risk_level = 'high'
        elif max_risk > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'has_urls': True,
            'num_urls': len(urls),
            'urls': url_details,
            'risk_level': risk_level,
            'max_risk_score': max_risk,
            'suspicious_indicators': list(set(suspicious_indicators))
        }


# Convenience function for quick feature extraction
def extract_url_features(message: str) -> Dict[str, float]:
    """Quick function to extract URL features from a message"""
    extractor = URLFeatureExtractor()
    return extractor.extract_message_url_features(message)


if __name__ == "__main__":
    # Test the URL feature extractor
    extractor = URLFeatureExtractor()
    
    test_messages = [
        "URGENT! Your account locked. Verify: bit.ly/secure123",
        "Your verification code is 123456. Valid for 10 minutes.",
        "Package delivery failed. Confirm: suspicious-domain.xyz/track",
        "Meeting at 3pm in conference room B",
        "Click here for amazing prizes: http://192.168.1.1/claim",
        "Your Amazon order has shipped. Track: https://www.amazon.com/track/12345"
    ]
    
    print("URL Feature Extraction Tests")
    print("=" * 80)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: {message[:60]}...")
        
        # Extract features
        features = extractor.extract_message_url_features(message)
        analysis = extractor.get_url_analysis(message)
        
        print(f"  Has URLs: {features['has_urls']}")
        print(f"  Risk Level: {analysis['risk_level']}")
        print(f"  Max Risk Score: {features['max_url_risk']:.3f}")
        
        if analysis['suspicious_indicators']:
            print(f"  Suspicious Indicators:")
            for indicator in analysis['suspicious_indicators']:
                print(f"    - {indicator}")
    
    print("\n" + "=" * 80)
    print("Feature extraction test complete!")
