# import re
# from .config import TRUSTED_DOMAINS, TRUSTED_PHRASES

# def check_whitelist(message: str) -> bool:
#     """
#     Returns a bool value:
#         True if the message is whitelisted, 
#         False otherwise.
#     """
#     message_lower = message.lower()

#     # 1. Check for whitelisted phrases
#     for phrase in TRUSTED_PHRASES:
#         if phrase in message_lower:
#             return True

#     # 2. Find all URLs in the message and then Check for whitelisted domains
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     urls = url_pattern.findall(message_lower)
    
#     for url in urls:
#         # Extract the domain from the URL
#         domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
#         if domain_match:
#             domain = domain_match.group(1)
#             if domain in TRUSTED_DOMAINS:
#                 return True        
#     return False

import re
import json
import os
import sys

# Get the directory of the current file (whitelisting.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the config.json file
CONFIG_FILE_PATH = os.path.join(current_dir, 'config.json')

# Load whitelist data from the JSON file once at module startup
try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        config_data = json.load(f)
    TRUSTED_DOMAINS = config_data.get('trusted_domains', [])
    TRUSTED_PHRASES = config_data.get('trusted_phrases', [])
    # print("Whitelist data loaded successfully.") # For debugging
except FileNotFoundError:
    print(f"Error: config.json not found at {CONFIG_FILE_PATH}. Using empty whitelists.")
    TRUSTED_DOMAINS = []
    TRUSTED_PHRASES = []
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from {CONFIG_FILE_PATH}. Using empty whitelists.")
    TRUSTED_DOMAINS = []
    TRUSTED_PHRASES = []
except Exception as e:
    print(f"An unexpected error occurred: {e}. Using empty whitelists.")
    TRUSTED_DOMAINS = []
    TRUSTED_PHRASES = []

def check_whitelist(message: str) -> bool:
    """
    Returns a bool value:
        True if the message is whitelisted, 
        False otherwise.
    """
    if not isinstance(message, str):
        return False
        
    message_lower = message.lower()

    # 1. Check for whitelisted phrases
    for phrase in TRUSTED_PHRASES:
        if phrase in message_lower:
            return True

    # 2. Find all URLs in the message and then check for whitelisted domains
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