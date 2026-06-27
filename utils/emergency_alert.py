"""
Emergency Alert System
Finds nearest hospital using OpenStreetMap (FREE - no API key needed!)
Sends SMS via Twilio and email via Gmail.
"""

import smtplib
import requests
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

CAMERA_LOCATION = {
    "address": "NH47, Tiruppur, Tamil Nadu, India",
    "lat": 11.1085,
    "lng": 77.3411,
}

TWILIO_ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID"
TWILIO_AUTH_TOKEN  = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_FROM_NUMBER = "+1234567890"
ALERT_TO_NUMBER    = "+91XXXXXXXXXX"

EMAIL_SENDER   = "your@gmail.com"
EMAIL_PASSWORD = "YOUR_APP_PASSWORD"
EMAIL_RECEIVER = "hospital@gmail.com"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Find Nearest Hospital — OpenStreetMap (FREE, no API key needed)
# ──────────────────────────────────────────────────────────────────────────────

def find_nearest_hospital(lat: float = None, lng: float = None) -> dict:
    lat = lat or CAMERA_LOCATION["lat"]
    lng = lng or CAMERA_LOCATION["lng"]

    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:5000,{lat},{lng});
          way["amenity"="hospital"](around:5000,{lat},{lng});
        );
        out center 1;
        """
        response = requests.post(overpass_url, data={"data": query}, timeout=15)
        data = response.json()

        if data.get("elements"):
            place = data["elements"][0]
            tags  = place.get("tags", {})
            hospital = {
                "name":    tags.get("name", "Nearby Hospital"),
                "address": tags.get("addr:full") or tags.get("addr:street", "Address not available"),
                "phone":   tags.get("phone", "112 (Emergency)"),
                "website": tags.get("website", "Not available"),
            }
            logger.info("Nearest hospital: %s", hospital["name"])
            return hospital

    except Exception as e:
        logger.error("OpenStreetMap lookup failed: %s", e)

    return {
        "name":    "Nearest Hospital",
        "address": CAMERA_LOCATION["address"],
        "phone":   "112 (Emergency)",
        "website": "Not available",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Send SMS via Twilio
# ──────────────────────────────────────────────────────────────────────────────

def send_sms_alert(accident_type: str, confidence: float, timestamp: float) -> bool:
    try:
        from twilio.rest import Client
    except ImportError:
        logger.error("Twilio not installed. Run: pip install twilio")
        return False

    hospital = find_nearest_hospital()
    time_str = datetime.now().strftime("%H:%M:%S")

    message = (
        f"ROAD ACCIDENT DETECTED\n"
        f"Type: {accident_type}\n"
        f"Confidence: {confidence:.0%}\n"
        f"Location: {CAMERA_LOCATION['address']}\n"
        f"Time: {time_str}\n"
        f"Nearest Hospital: {hospital['name']}\n"
        f"Address: {hospital['address']}\n"
        f"Phone: {hospital['phone']}\n"
        f"Please respond immediately!"
    )

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=ALERT_TO_NUMBER)
        logger.info("SMS sent to %s", ALERT_TO_NUMBER)
        return True
    except Exception as e:
        logger.error("SMS failed: %s", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 3. Send Email Alert with Screenshot
# ──────────────────────────────────────────────────────────────────────────────

def send_email_alert(accident_type: str, confidence: float, timestamp_s: float, screenshot_path: str = None) -> bool:
    hospital = find_nearest_hospital()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = MIMEMultipart()
    msg["Subject"] = f"ROAD ACCIDENT DETECTED - {accident_type}"
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER

    body = f"""
    <html><body style="font-family:Arial,sans-serif;">
      <div style="background:#e74c3c;color:white;padding:15px;border-radius:8px;">
        <h2>Road Accident Detected</h2>
      </div>
      <div style="background:white;padding:20px;margin-top:10px;border-radius:8px;">
        <h3>Accident Details</h3>
        <table>
          <tr><td><b>Type</b></td><td>{accident_type}</td></tr>
          <tr><td><b>Confidence</b></td><td>{confidence:.0%}</td></tr>
          <tr><td><b>Time</b></td><td>{time_str}</td></tr>
          <tr><td><b>Location</b></td><td>{CAMERA_LOCATION['address']}</td></tr>
        </table>
        <h3>Nearest Hospital</h3>
        <table>
          <tr><td><b>Name</b></td><td>{hospital['name']}</td></tr>
          <tr><td><b>Address</b></td><td>{hospital['address']}</td></tr>
          <tr><td><b>Phone</b></td><td>{hospital['phone']}</td></tr>
        </table>
        <p style="color:red;"><b>Please dispatch emergency services immediately.</b></p>
      </div>
    </body></html>
    """
    msg.attach(MIMEText(body, "html"))

    if screenshot_path and Path(screenshot_path).exists():
        with open(screenshot_path, "rb") as f:
            img = MIMEImage(f.read(), name=Path(screenshot_path).name)
            msg.attach(img)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Email sent to %s", EMAIL_RECEIVER)
        return True
    except Exception as e:
        logger.error("Email failed: %s", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 4. Master function — call this when accident is detected
# ──────────────────────────────────────────────────────────────────────────────

def trigger_emergency_response(accident_type: str, confidence: float, timestamp_s: float, screenshot_path: str = None):
    logger.warning("Triggering emergency response: %s", accident_type)

    hospital = find_nearest_hospital()

    print("\n" + "="*55)
    print("  EMERGENCY RESPONSE TRIGGERED")
    print("="*55)
    print(f"  Accident  : {accident_type}")
    print(f"  Confidence: {confidence:.0%}")
    print(f"  Location  : {CAMERA_LOCATION['address']}")
    print(f"  Hospital  : {hospital['name']}")
    print(f"  Address   : {hospital['address']}")
    print(f"  Phone     : {hospital['phone']}")
    print("="*55 + "\n")

    sms_sent   = send_sms_alert(accident_type, confidence, timestamp_s)
    email_sent = send_email_alert(accident_type, confidence, timestamp_s, screenshot_path)

    return {
        "hospital":   hospital,
        "sms_sent":   sms_sent,
        "email_sent": email_sent,
    }
EOF
echo "done"