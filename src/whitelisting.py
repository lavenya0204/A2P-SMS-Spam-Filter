import re
from config import TRUSTED_DOMAINS, TRUSTED_PHRASES

def check_whitelist(message: str) -> bool:
    """
    Returns a bool value:
        True if the message is whitelisted, 
        False otherwise.
    """
    message_lower = message.lower()

    # 1. Check for whitelisted phrases
    for phrase in TRUSTED_PHRASES:
        if phrase in message_lower:
            return True

    # 2. Find all URLs in the message and then Check for whitelisted domains
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = url_pattern.findall(message_lower)
    
    for url in urls:
        # Extract the domain from the URL
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if domain_match:
            domain = domain_match.group(1)
            if domain in TRUSTED_DOMAINS:
                return True        
    return False