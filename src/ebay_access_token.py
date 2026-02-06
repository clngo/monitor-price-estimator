import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# Credentials from .env
EBAY_APP_ID = os.getenv("EBAY_APP_ID")  # Client ID
EBAY_CERT_ID = os.getenv("EBAY_CERT_ID")  # Client Secret

TOKEN_URL_PRODUCTION = "https://api.ebay.com/identity/v1/oauth2/token"
SCOPE = "https://api.ebay.com/oauth/api_scope"


def generate_ebay_oauth_token():
    if not EBAY_APP_ID or not EBAY_CERT_ID:
        raise ValueError("EBAY_APP_ID or EBAY_CERT_ID not set in environment variables.")

    # Encode client_id:client_secret in Base64
    credentials = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_credentials}",
    }

    data = {
        "grant_type": "client_credentials",
        "scope": SCOPE,
    }

    response = requests.post(TOKEN_URL_PRODUCTION, headers=headers, data=data, timeout=30)
    response.raise_for_status()

    return response.json()


def update_env_token(env_path: str, token: str):
    """
    Updates the EBAY_OAUTH_TOKEN line in the .env file.
    If the line does not exist, it will be added.
    """
    lines = []
    found = False

    # Read existing .env
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    # Update EBAY_OAUTH_TOKEN line
    for i, line in enumerate(lines):
        if line.startswith("EBAY_OAUTH_TOKEN="):
            lines[i] = f"EBAY_OAUTH_TOKEN={token}\n"
            found = True
            break

    # If not found, append it
    if not found:
        lines.append(f"EBAY_OAUTH_TOKEN={token}\n")

    # Write back
    with open(env_path, "w") as f:
        f.writelines(lines)

    print(f"Updated EBAY_OAUTH_TOKEN in {env_path}")


if __name__ == "__main__":
    # Generate OAuth token
    token_response = generate_ebay_oauth_token()

    access_token = token_response.get("access_token")
    expires_in = token_response.get("expires_in")

    print("Access Token:", access_token)
    print("Expires In (seconds):", expires_in)

    # Update your .env file directly
    env_file_path = ".env"  # adjust path if your .env is somewhere else
    update_env_token(env_file_path, access_token)
