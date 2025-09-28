# core/schedule.py
from datetime import datetime, timedelta


def make_ics(title, start_dt, duration_min=30, location=""):
    dtfmt = "%Y%m%dT%H%M%S"
    end_dt = start_dt + timedelta(minutes=duration_min)
    return f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART:{start_dt.strftime(dtfmt)}
DTEND:{end_dt.strftime(dtfmt)}
SUMMARY:{title}
LOCATION:{location}
END:VEVENT
END:VCALENDAR
"""
