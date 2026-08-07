"""
Google Calendar integration service.
Lists upcoming meetings and helps prepare for them with financial context.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from googleapiclient.discovery import build

from services.google.gmail import _build_credentials, SCOPES


async def get_upcoming_events(token_data: Dict, days: int = 7) -> str:
    """
    Get upcoming calendar events with financial context.
    Identifies company-related meetings and adds relevant context.
    """
    try:
        creds = await asyncio.to_thread(_build_credentials, token_data)
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)

        now = datetime.now(timezone.utc).isoformat()
        end = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()

        events_result = await asyncio.to_thread(
            lambda: service.events().list(
                calendarId="primary",
                timeMin=now,
                timeMax=end,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime"
            ).execute()
        )

        events = events_result.get("items", [])

        if not events:
            return f"No calendar events found in the next {days} days."

        lines = [f"<b>📅 Upcoming Calendar Events ({days} days)</b>\n"]

        for event in events:
            title = event.get("summary", "Untitled")
            start = event["start"].get("dateTime", event["start"].get("date", ""))
            location = event.get("location", "")
            attendees = event.get("attendees", [])
            attendee_count = len(attendees)

            # Parse and format date
            try:
                if "T" in start:
                    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    formatted_time = dt.strftime("%a %b %d, %I:%M %p")
                else:
                    formatted_time = start
            except Exception:
                formatted_time = start

            line = f"• <b>{title}</b>\n  🕐 {formatted_time}"
            if location:
                line += f"\n  📍 {location}"
            if attendee_count > 0:
                line += f"\n  👥 {attendee_count} attendees"

            lines.append(line)

        return "\n\n".join(lines)

    except Exception as e:
        if "invalid_grant" in str(e).lower():
            return "Your Google Calendar connection has expired. Please reconnect."
        return f"Could not retrieve calendar events: {str(e)}"


async def get_todays_meetings(token_data: Dict) -> List[Dict]:
    """Get today's meetings as a list of dicts (for morning briefing context)."""
    try:
        creds = await asyncio.to_thread(_build_credentials, token_data)
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)

        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0).isoformat()
        end_of_day = now.replace(hour=23, minute=59, second=59).isoformat()

        events_result = await asyncio.to_thread(
            lambda: service.events().list(
                calendarId="primary",
                timeMin=start_of_day,
                timeMax=end_of_day,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime"
            ).execute()
        )

        return events_result.get("items", [])

    except Exception:
        return []
