"""
Google Gmail integration service.
OAuth2 flow + email search and summarization for finance professionals.
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from config import settings

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# OAuth client config built from environment variables
def _get_client_config() -> Dict:
    return {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uris": [settings.google_redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


def create_oauth_flow() -> Flow:
    """Create a Google OAuth2 flow for the consent screen."""
    return Flow.from_client_config(
        _get_client_config(),
        scopes=SCOPES,
        redirect_uri=settings.google_redirect_uri
    )


def get_authorization_url(state: str) -> str:
    """
    Generate the Google OAuth authorization URL.
    State is the user's Telegram ID so we know who to update after callback.
    """
    flow = create_oauth_flow()
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=state
    )
    return auth_url


async def exchange_code_for_tokens(code: str) -> Dict[str, Any]:
    """Exchange OAuth authorization code for access + refresh tokens."""
    flow = create_oauth_flow()
    await asyncio.to_thread(flow.fetch_token, code=code)
    credentials = flow.credentials
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes) if credentials.scopes else SCOPES,
    }


def _build_credentials(token_data: Dict) -> Credentials:
    """Build Google Credentials object from stored token dict."""
    creds = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_data.get("client_id", settings.google_client_id),
        client_secret=token_data.get("client_secret", settings.google_client_secret),
        scopes=token_data.get("scopes", SCOPES),
    )

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return creds


async def search_emails(token_data: Dict, query: str, max_results: int = 10) -> str:
    """
    Search Gmail for emails matching a query.
    Returns formatted summaries of matching threads.
    """
    try:
        creds = await asyncio.to_thread(_build_credentials, token_data)
        service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)

        # Search messages
        results = await asyncio.to_thread(
            lambda: service.users().messages().list(
                userId="me",
                q=query,
                maxResults=max_results
            ).execute()
        )

        messages = results.get("messages", [])
        if not messages:
            return f"No emails found matching: <i>{query}</i>"

        # Get snippet for each message
        email_summaries = []
        for msg in messages[:5]:
            msg_data = await asyncio.to_thread(
                lambda m=msg: service.users().messages().get(
                    userId="me",
                    id=m["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From", "Date"]
                ).execute()
            )

            headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject", "No subject")[:80]
            sender = headers.get("From", "Unknown")[:50]
            snippet = msg_data.get("snippet", "")[:100]

            email_summaries.append(f"• <b>{subject}</b>\n  From: {sender}\n  {snippet}...")

        lines = [f"<b>📧 Emails matching '{query}'</b>\n"]
        lines.extend(email_summaries)
        return "\n\n".join(lines)

    except Exception as e:
        if "invalid_grant" in str(e).lower():
            return "Your Gmail connection has expired. Please reconnect: send 'connect gmail'"
        return f"Could not search emails: {str(e)}"


async def get_email_count(token_data: Dict, query: str) -> int:
    """Get count of emails matching a query."""
    try:
        creds = await asyncio.to_thread(_build_credentials, token_data)
        service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)
        results = await asyncio.to_thread(
            lambda: service.users().messages().list(
                userId="me", q=query, maxResults=1
            ).execute()
        )
        return results.get("resultSizeEstimate", 0)
    except Exception:
        return 0
