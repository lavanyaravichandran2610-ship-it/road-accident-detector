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

CAMERA_LOCATION = {
    "name": "NH47 Junction, Tiruppur, Tamil Nadu",
    "lat": 11.1085,
    "lng": 77.3411,
}

try:
    import streamlit as st
    EMAIL_SENDER = st.secrets.get("EMAIL_SENDER", "your@gmail.com")
    EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "YOUR_APP_PASSWORD")
    NTFY_TOPIC = st.secrets.get("NTFY_TOPIC", "road-accident-alert")
    CONTROL_ROOM_EMAIL = st.secrets.get("CONTROL_ROOM_EMAIL", "your@gmail.com")
except Exception:
    EMAIL_SENDER = "your@gmail.com"
    EMAIL_PASSWORD = "YOUR_APP_PASSWORD"
    NTFY_TOPIC = "road-accident-alert"
    CONTROL_ROOM_EMAIL = "your@gmail.com"

EMERGENCY_NUMBERS = {
    "ambulance": "108",
    "police": "100",
    "fire": "101",
    "emergency": "112",
}


def calculate_distance_km(lat1, lng1, lat2, lng2):
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lng / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def get_live_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=10)
        data = r.json()
        return {
            "lat": data.get("latitude", CAMERA_LOCATION["lat"]),
            "lng": data.get("longitude", CAMERA_LOCATION["lng"]),
            "city": data.get("city", "Tiruppur"),
            "region": data.get("region", "Tamil Nadu"),
            "country": data.get("country_name", "India"),
        }
    except Exception:
        return {
            "lat": CAMERA_LOCATION["lat"],
            "lng": CAMERA_LOCATION["lng"],
            "city": "Tiruppur",
            "region": "Tamil Nadu",
            "country": "India",
        }


def find_nearest_place(place_type, lat, lng):
    place_label = "hospital" if place_type == "hospital" else "police station"
    emergency_phone = EMERGENCY_NUMBERS["ambulance"] if place_type == "hospital" else EMERGENCY_NUMBERS["police"]

    result = {
        "name": f"Nearest {place_label.title()}",
        "address": "See navigation link",
        "phone": emergency_phone,
        "distance_km": None,
        "distance_text": "Calculating...",
        "maps_nav": f"https://www.google.com/maps/search/{place_label.replace(' ', '+')}/@{lat},{lng},15z",
        "maps_dir": f"https://www.google.com/maps/dir/{lat},{lng}/{place_label.replace(' ', '+')}",
        "auto_detected": False,
    }

    try:
        headers = {"User-Agent": "RoadAccidentDetectionSystem/1.0"}
        search_q = "hospital" if place_type == "hospital" else "police"
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={search_q}&format=json&limit=1"
            f"&lat={lat}&lon={lng}&addressdetails=1&countrycodes=in"
        )
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                place = data[0]
                place_lat = float(place.get("lat", lat))
                place_lng = float(place.get("lon", lng))
                dist = calculate_distance_km(lat, lng, place_lat, place_lng)
                name = place.get("display_name", result["name"]).split(",")[0]
                addr = ", ".join(place.get("display_name", "").split(",")[:3])

                result.update({
                    "name": name,
                    "address": addr,
                    "distance_km": dist,
                    "distance_text": f"{dist} km away",
                    "maps_nav": f"https://www.google.com/maps/search/{place_label.replace(' ', '+')}/@{place_lat},{place_lng},17z",
                    "maps_dir": f"https://www.google.com/maps/dir/{lat},{lng}/{place_lat},{place_lng}",
                    "auto_detected": True,
                })
                logger.info("Auto-detected %s: %s (%s km)", place_label, name, dist)

    except Exception as e:
        logger.warning("Auto-detect %s failed: %s", place_label, e)

    return result


def send_ntfy(accident_type, confidence, location, hospital, police, maps_link, time_str):
    try:
        urgency = "CRITICAL" if confidence > 0.85 else "MODERATE"
        message = (
            f"Type      : {accident_type}\n"
            f"Severity  : {urgency}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Location  : {location['city']}, {location['region']}\n"
            f"Time      : {time_str}\n\n"
            f"NEAREST HOSPITAL\n"
            f"{hospital['name']}\n"
            f"{hospital['distance_text']}\n"
            f"Dispatch  : Call {EMERGENCY_NUMBERS['ambulance']}\n\n"
            f"NEAREST POLICE\n"
            f"{police['name']}\n"
            f"{police['distance_text']}\n"
            f"Dispatch  : Call {EMERGENCY_NUMBERS['police']}\n\n"
            f"Accident Map: {maps_link}"
        )
        r = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={
                "Title": f"{urgency} ACCIDENT: {accident_type}",
                "Priority": "urgent",
                "Tags": "rotating_light,ambulance,police_car",
                "Click": maps_link,
            },
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        logger.error("Ntfy error: %s", e)
        return False


def send_control_room_email(accident_type, confidence, timestamp_s, hospital, police,
                              location, maps_link, time_str, screenshot_path=None):
    try:
        urgency = "CRITICAL" if confidence > 0.85 else "MODERATE"
        urgency_color = "#c0392b" if urgency == "CRITICAL" else "#e67e22"

        msg = MIMEMultipart()
        msg["Subject"] = f"[{urgency}] ROAD ACCIDENT - {accident_type} - Auto-Dispatch Alert"
        msg["From"] = EMAIL_SENDER
        msg["To"] = CONTROL_ROOM_EMAIL

        hospital_dist = hospital.get("distance_text", "Unknown distance")
        police_dist = police.get("distance_text", "Unknown distance")

        body = f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">

          <div style="background:{urgency_color};color:white;padding:20px;border-radius:8px;">
            <h2 style="margin:0;">🚨 TRAFFIC ACCIDENT DETECTED — {urgency}</h2>
            <p style="margin:5px 0 0;">Automated 24/7 CCTV Monitoring System</p>
          </div>

          <div style="background:white;padding:20px;margin-top:10px;border-radius:8px;">
            <p><b>Action required:</b> Dispatch emergency services to the location below.
            This alert was generated automatically — no manual report was filed.</p>

            <h3 style="color:{urgency_color};border-bottom:2px solid {urgency_color};padding-bottom:5px;">
              Accident Details
            </h3>
            <table style="border-collapse:collapse;width:100%;">
              <tr style="background:#f8f8f8;">
                <td style="padding:10px;border:1px solid #ddd;width:35%;"><b>Type</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{accident_type}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Severity</b></td>
                <td style="padding:10px;border:1px solid #ddd;color:{urgency_color};font-weight:bold;">{urgency} ({confidence:.0%} confidence)</td>
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
                <td style="padding:10px;border:1px solid #ddd;"><b>Camera</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{CAMERA_LOCATION['name']}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Accident Location</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <a href="{maps_link}" style="background:{urgency_color};color:white;padding:6px 12px;border-radius:4px;text-decoration:none;">
                    Open in Google Maps
                  </a>
                </td>
              </tr>
            </table>

            <h3 style="color:#27ae60;border-bottom:2px solid #27ae60;padding-bottom:5px;margin-top:20px;">
              🏥 Dispatch Ambulance — Nearest Hospital (Auto-Detected)
            </h3>
            <table style="border-collapse:collapse;width:100%;">
              <tr style="background:#f0fff4;">
                <td style="padding:10px;border:1px solid #ddd;width:35%;"><b>Facility</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{hospital['name']}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Address</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{hospital['address']}</td>
              </tr>
              <tr style="background:#f0fff4;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Distance from accident</b></td>
                <td style="padding:10px;border:1px solid #ddd;color:#27ae60;font-weight:bold;">{hospital_dist}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Call Ambulance</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <b style="color:#c0392b;font-size:1.3em;">{EMERGENCY_NUMBERS['ambulance']}</b>
                </td>
              </tr>
              <tr style="background:#f0fff4;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Navigate</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <a href="{hospital['maps_dir']}" style="background:#27ae60;color:white;padding:6px 12px;border-radius:4px;text-decoration:none;">
                    Get Directions
                  </a>
                </td>
              </tr>
            </table>

            <h3 style="color:#2980b9;border-bottom:2px solid #2980b9;padding-bottom:5px;margin-top:20px;">
              🚔 Dispatch Police — Nearest Station (Auto-Detected)
            </h3>
            <table style="border-collapse:collapse;width:100%;">
              <tr style="background:#eff6ff;">
                <td style="padding:10px;border:1px solid #ddd;width:35%;"><b>Station</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{police['name']}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Address</b></td>
                <td style="padding:10px;border:1px solid #ddd;">{police['address']}</td>
              </tr>
              <tr style="background:#eff6ff;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Distance from accident</b></td>
                <td style="padding:10px;border:1px solid #ddd;color:#2980b9;font-weight:bold;">{police_dist}</td>
              </tr>
              <tr>
                <td style="padding:10px;border:1px solid #ddd;"><b>Call Police</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <b style="color:#c0392b;font-size:1.3em;">{EMERGENCY_NUMBERS['police']}</b>
                </td>
              </tr>
              <tr style="background:#eff6ff;">
                <td style="padding:10px;border:1px solid #ddd;"><b>Navigate</b></td>
                <td style="padding:10px;border:1px solid #ddd;">
                  <a href="{police['maps_dir']}" style="background:#2980b9;color:white;padding:6px 12px;border-radius:4px;text-decoration:none;">
                    Get Directions
                  </a>
                </td>
              </tr>
            </table>

            <br>
            <div style="background:#ffeeba;padding:15px;border-radius:8px;border-left:5px solid {urgency_color};margin-top:20px;">
              <p style="margin:0;color:#856404;">
                <b>⚠️ Traffic Control Note:</b> Please alert upstream traffic signals/cameras
                to redirect or warn approaching vehicles and prevent lateral/secondary collisions
                at this location.<br><br>
                Accident screenshot is attached.
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

        logger.info("Control room email sent to %s", CONTROL_ROOM_EMAIL)
        return True

    except Exception as e:
        logger.error("Control room email failed: %s", e)
        return False


def trigger_emergency_response(accident_type, confidence, timestamp_s, screenshot_path=None):
    print("\n" + "=" * 55)
    print("   AUTOMATED EMERGENCY DISPATCH TRIGGERED")
    print("=" * 55)
    print(f"  Accident  : {accident_type}")
    print(f"  Confidence: {confidence:.0%}")

    location = get_live_location()
    lat = location["lat"]
    lng = location["lng"]
    print(f"  Location  : {location['city']}, {location['region']}")

    hospital = find_nearest_place("hospital", lat, lng)
    police = find_nearest_place("police", lat, lng)
    print(f"  Hospital  : {hospital['name']} ({hospital['distance_text']})")
    print(f"  Police    : {police['name']} ({police['distance_text']})")

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    maps_link = f"https://maps.google.com/?q={lat},{lng}"

    ntfy_ok = send_ntfy(accident_type, confidence, location, hospital, police, maps_link, time_str)
    print(f"  Ntfy      : {'OK' if ntfy_ok else 'FAILED'}")

    email_ok = send_control_room_email(
        accident_type, confidence, timestamp_s,
        hospital, police, location, maps_link, time_str, screenshot_path,
    )
    print(f"  Email     : {'OK' if email_ok else 'FAILED'}")
    print("=" * 55)

    return {
        "hospital": hospital,
        "police": police,
        "location": location,
        "ntfy": ntfy_ok,
        "email": email_ok,
    }
