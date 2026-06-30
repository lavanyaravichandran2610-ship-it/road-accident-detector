"""
Emergency Alert System
Finds nearest hospital using OpenStreetMap (FREE - no API key needed!)
Sends SMS via Twilio and email via Gmail.
"""

import smtplib
import requests
import logging
import math
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CAMERA_LOCATION = {
    "name": "NH47 Junction, Tiruppur, Tamil Nadu",
    "lat":  11.1085,
    "lng":  77.3411,
}

try:
    import streamlit as st
    EMAIL_SENDER   = st.secrets.get("EMAIL_SENDER",   "your@gmail.com")
    EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "YOUR_APP_PASSWORD")
    NTFY_TOPIC     = st.secrets.get("NTFY_TOPIC",     "road-accident-alert")
except Exception:
    EMAIL_SENDER   = "your@gmail.com"
    EMAIL_PASSWORD = "YOUR_APP_PASSWORD"
    NTFY_TOPIC     = "road-accident-alert"

# Your personal email to always receive alerts
PERSONAL_EMAIL = "lavanyaravichandran2610@gmail.com"

# India emergency numbers — always work nationwide
EMERGENCY_NUMBERS = {
    "ambulance": "108",
    "police":    "100",
    "fire":      "101",
    "emergency": "112",
}

# ==============================================================================
# DISTANCE CALCULATOR
# ==============================================================================

def calculate_distance_km(lat1, lng1, lat2, lng2):
    """Calculate distance in km between two GPS coordinates."""
    R = 6371  # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lng / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)

# ==============================================================================
# GET LIVE LOCATION FROM IP
# ==============================================================================

def get_live_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=10)
        data = r.json()
        return {
            "lat":     data.get("latitude",     CAMERA_LOCATION["lat"]),
            "lng":     data.get("longitude",    CAMERA_LOCATION["lng"]),
            "city":    data.get("city",         "Tiruppur"),
            "region":  data.get("region",       "Tamil Nadu"),
            "country": data.get("country_name", "India"),
        }
    except Exception:
        return {
            "lat":     CAMERA_LOCATION["lat"],
            "lng":     CAMERA_LOCATION["lng"],
            "city":    "Tiruppur",
            "region":  "Tamil Nadu",
            "country": "India",
        }

# ==============================================================================
# AUTO DETECT NEAREST HOSPITAL AND POLICE
# ==============================================================================

def find_nearest_place(place_type, lat, lng):
    """
    Auto-detects nearest hospital or police station.
    Uses Nominatim (OpenStreetMap) - free, no API key needed.
    Returns name, distance, phone, and navigation links.
    """
    place_label = "hospital" if place_type == "hospital" else "police station"
    emergency_phone = "108" if place_type == "hospital" else "100"

    # Default fallback
    result = {
        "name":       f"Nearest {place_label.title()} (Auto-detected)",
        "address":    "See map link below",
        "phone":      emergency_phone,
        "distance":   "Unknown",
        "maps_nav":   f"https://www.google.com/maps/search/{place_label.replace(' ', '+')}/@{lat},{lng},15z",
        "maps_dir":   f"https://www.google.com/maps/dir/{lat},{lng}/{place_label.replace(' ', '+')}",
    }

    try:
        headers = {"User-Agent": "RoadAccidentDetectionSystem/1.0"}
        search_q = "hospital" if place_type == "hospital" else "police"
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={search_q}"
            f"&format=json"
            f"&limit=1"
            f"&lat={lat}&lon={lng}"
            f"&addressdetails=1"
            f"&countrycodes=in"
        )
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                place     = data[0]
                place_lat = float(place.get("lat", lat))
                place_lng = float(place.get("lon", lng))
                dist      = calculate_distance_km(lat, lng, place_lat, place_lng)
                name      = place.get("display_name", result["name"]).split(",")[0]
                addr      = ", ".join(place.get("display_name", "").split(",")[:3])

                result.update({
                    "name":     name,
                    "address":  addr,
                    "distance": f"{dist} km away",
                    "maps_nav": f"https://www.google.com/maps/search/{place_label.replace(' ', '+')}/@{place_lat},{place_lng},17z",
                    "maps_dir": f"https://www.google.com/maps/dir/{lat},{lng}/{place_lat},{place_lng}",
                })
                logger.info("Found %s: %s (%s km)", place_label, name, dist)

    except Exception as e:
        logger.warning("Auto-detect %s failed: %s", place_label, e)

    return result

# ==============================================================================
# SEND NTFY NOTIFICATION
# ==============================================================================

def send_ntfy(accident_type, confidence, location, hospital, police, maps_link, time_str):
    try:
        message = (
            f"Type      : {accident_type}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Location  : {location['city']}, {location['region']}\n"
            f"Time      : {time_str}\n\n"
            f"HOSPITAL  : {hospital['name']}\n"
            f"Distance  : {hospital['distance']}\n"
            f"Call      : {EMERGENCY_NUMBERS['ambulance']} (Ambulance)\n\n"
            f"POLICE    : {police['name']}\n"
            f"Distance  : {police['distance']}\n"
            f"Call      : {EMERGENCY_NUMBERS['police']} (Police)\n\n"
            f"Map       : {maps_link}"
        )
        r = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={
                "Title":    f"ACCIDENT: {accident_type}",
                "Priority": "urgent",
                "Tags":     "rotating_light,ambulance",
                "Click":    maps_link,
            },
            timeout=10,
        )
        if r.status_code == 200:
            logger.info("Ntfy sent")
            return True
        return False
    except Exception as e:
        logger.error("Ntfy error: %s", e)
        return False

# ==============================================================================
# SEND EMAIL
# ==============================================================================

def send_email(to_email, contact_name, accident_type, confidence,
               timestamp_s, hospital, police, location,
               maps_link, time_str, screenshot_path=None):
    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"ROAD ACCIDENT DETECTED - {accident_type}"
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = to_email

        body = f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">

          <div style="background:#e74c3c;color:white;padding:20px;border-radius:8px;">
            <h2 style="margin:0;">🚨 Road Accident Detected</h2>
            <p style="margin:5px 0 0;">Automated CCTV Monitoring Alert</p>
          </div>

          <div style="background:white;padding:20px;margin-top:10px;border-radius:8px;">
            <p>Dear {contact_name},</p>
            <p><b style="color:#e74c3c;">A road accident has been detected. Please respond immediately.</b></p>

            <h3 style="color:#e74c3c;border-bottom:2px solid #e74c3c;padding-bottom:5px;">
              Accident Details
            </h3>
            <table style="border-collapse:collapse;width:100%;">
              <tr style="background:#f8f8f8;">
                <td style="padding:10px;border:1px solid #ddd;width:35%;"><b>Type</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{accident_type}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Confidence</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{confidence:.0%}</td>
              </tr>
              <tr style="background:#f8f8f8;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Time</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{time_str}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>City</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{location['city']}, {location['region']}</td>
              </tr>
              <tr style="background:#f8f8f8;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Camera Location</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{CAMERA_LOCATION['name']}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Accident Location</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <a href="{maps_link}" style="background:#e74c3c;color:white;padding:6px 12px;border-radius:4px;text-decoration:none;">
                    Open in Google Maps
                  </a>
                </td>
              </tr>
            </table>

            <h3 style="color:#27ae60;border-bottom:2px solid #27ae60;padding-bottom:5px;margin-top:20px;">
              🏥 Nearest Hospital (Auto-Detected)
            </h3>
            <table style="border-collapse:collapse;width:100%;">
              <tr style="background:#f0fff4;">
                <td style="padding:10px;border:1px solid #ddd;width:35%;"><b>Name</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{hospital['name']}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Address</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{hospital['address']}</td>
              </tr>
              <tr style="background:#f0fff4;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Distance</b></td>
                <td style="padding:10px;border:1px solid #ddd;color:#27ae60;font-weight:bold;">{hospital['distance']}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Emergency Call</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <b style="color:#e74c3c;font-size:1.2em;">{EMERGENCY_NUMBERS['ambulance']}</b> (Ambulance)
                  &nbsp;|&nbsp;
                  <b style="color:#e74c3c;font-size:1.2em;">{EMERGENCY_NUMBERS['emergency']}</b> (National Emergency)
                </td>
              </tr>
              <tr style="background:#f0fff4;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Navigate</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <a href="{hospital['maps_dir']}" style="background:#27ae60;color:white;padding:6px 12px;border-radius:4px;text-decoration:none;">
                    Get Directions to Hospital
                  </a>
                </td>
              </tr>
            </table>

            <h3 style="color:#2980b9;border-bottom:2px solid #2980b9;padding-bottom:5px;margin-top:20px;">
              🚔 Nearest Police Station (Auto-Detected)
            </h3>
            <table style="border-collapse:collapse;width:100%;">
              <tr style="background:#eff6ff;">
                <td style="padding:10px;border:1px solid #ddd;width:35%;"><b>Name</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{police['name']}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Address</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{police['address']}</td>
              </tr>
              <tr style="background:#eff6ff;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Distance</b></td>
                <td style="padding:10px;border:1px solid #ddd;color:#2980b9;font-weight:bold;">{police['distance']}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Emergency Call</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <b style="color:#2980b9;font-size:1.2em;">{EMERGENCY_NUMBERS['police']}</b> (Police)
                  &nbsp;|&nbsp;
                  <b style="color:#2980b9;font-size:1.2em;">{EMERGENCY_NUMBERS['emergency']}</b> (National Emergency)
                </td>
              </tr>
              <tr style="background:#eff6ff;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Navigate</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <a href="{police['maps_dir']}" style="background:#2980b9;color:white;padding:6px 12px;border-radius:4px;text-decoration:none;">
                    Get Directions to Police Station
                  </a>
                </td>
              </tr>
            </table>

            <br>
            <div style="background:#ffeeba;padding:15px;border-radius:8px;border-left:5px solid #e74c3c;margin-top:20px;">
              <p style="margin:0;color:#856404;font-size:1.05em;">
                <b>⚠️ Immediate Action Required:</b><br>
                Call <b>{EMERGENCY_NUMBERS['ambulance']}</b> for ambulance &nbsp;|&nbsp;
                Call <b>{EMERGENCY_NUMBERS['police']}</b> for police &nbsp;|&nbsp;
                Call <b>{EMERGENCY_NUMBERS['emergency']}</b> for national emergency<br><br>
                Accident screenshot is attached to this email.
              </p>
            </div>
          </div>

        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as f:
                img = MIMEImage(f.read(), name=Path(screenshot_path).name)
                msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        logger.info("Email sent to %s", to_email)
        return True

    except Exception as e:
        logger.error("Email failed to %s: %s", to_email, e)
        return False

# ==============================================================================
# MASTER FUNCTION
# ==============================================================================

def trigger_emergency_response(accident_type, confidence, timestamp_s, screenshot_path=None):
    print("\n" + "=" * 55)
    print("   EMERGENCY RESPONSE TRIGGERED")
    print("=" * 55)
    print(f"  Accident  : {accident_type}")
    print(f"  Conf      : {confidence:.0%}")

    location = get_live_location()
    lat = location["lat"]
    lng = location["lng"]
    print(f"  Location  : {location['city']}, {location['region']}")

    hospital = find_nearest_place("hospital", lat, lng)
    police   = find_nearest_place("police",   lat, lng)
    print(f"  Hospital  : {hospital['name']} ({hospital['distance']})")
    print(f"  Police    : {police['name']} ({police['distance']})")

    time_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    maps_link = f"https://maps.google.com/?q={lat},{lng}"

    ntfy_ok = send_ntfy(
        accident_type, confidence, location,
        hospital, police, maps_link, time_str,
    )
    print(f"  Ntfy      : {'OK' if ntfy_ok else 'FAILED'}")

    email_ok = send_email(
        to_email=PERSONAL_EMAIL,
        contact_name="Emergency Contact",
        accident_type=accident_type,
        confidence=confidence,
        timestamp_s=timestamp_s,
        hospital=hospital,
        police=police,
        location=location,
        maps_link=maps_link,
        time_str=time_str,
        screenshot_path=screenshot_path,
    )
    print(f"  Email     : {'OK' if email_ok else 'FAILED'}")
    print("=" * 55)

    return {
        "hospital": hospital,
        "police":   police,
        "location": location,
        "ntfy":     ntfy_ok,
        "email":    email_ok,
    }
