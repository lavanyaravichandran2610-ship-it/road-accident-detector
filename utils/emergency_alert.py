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

    # Twilio - OPTIONAL. Leave blank in secrets to disable auto-call entirely.
    TWILIO_ACCOUNT_SID = st.secrets.get("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN  = st.secrets.get("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER = st.secrets.get("TWILIO_FROM_NUMBER", "")
    TWILIO_CALL_TO     = st.secrets.get("TWILIO_CALL_TO", "")  # must be a verified number on trial accounts
except Exception:
    EMAIL_SENDER = "your@gmail.com"
    EMAIL_PASSWORD = "YOUR_APP_PASSWORD"
    NTFY_TOPIC = "road-accident-alert"
    CONTROL_ROOM_EMAIL = "your@gmail.com"
    TWILIO_ACCOUNT_SID = ""
    TWILIO_AUTH_TOKEN  = ""
    TWILIO_FROM_NUMBER = ""
    TWILIO_CALL_TO     = ""

EMERGENCY_NUMBERS = {
    "ambulance": "108",
    "police": "100",
    "fire": "101",
    "emergency": "112",
}

AUTO_CALL_ENABLED = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and TWILIO_CALL_TO)


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
    """
    Returns the FIXED camera location (CAMERA_LOCATION).
    We deliberately do NOT use IP-based geolocation here, because on
    cloud hosting (Streamlit Cloud, Render, etc.) the server's IP
    location has nothing to do with where the physical camera is
    installed - it would return the data center's location instead,
    causing wildly wrong hospital/police results.

    CAMERA_LOCATION is set explicitly per camera (see CAMERA_REGISTRY
    in app.py) using its real installed GPS coordinates.
    """
    city = CAMERA_LOCATION["name"].split(",")[0].strip()
    return {
        "lat": CAMERA_LOCATION["lat"],
        "lng": CAMERA_LOCATION["lng"],
        "city": city,
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


def make_auto_call(accident_type, confidence, location, hospital, police):
    """
    OPTIONAL Twilio voice call - only runs if all Twilio secrets are configured.
    On a free Twilio trial, TWILIO_CALL_TO MUST be a verified number
    (Twilio Console -> Phone Numbers -> Verified Caller IDs).
    """
    if not AUTO_CALL_ENABLED:
        logger.info("Auto-call skipped - Twilio not configured")
        return False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        urgency = "critical" if confidence > 0.85 else "moderate"

        twiml = (
            "<Response>"
            "<Say voice='alice' loop='2'>"
            f"Automated traffic accident alert. "
            f"A {urgency} severity {accident_type} has been detected "
            f"at {location['city']}, {location['region']}. "
            f"Confidence level {int(confidence * 100)} percent. "
            f"Nearest hospital is {hospital['name']}, "
            f"{hospital['distance_text']}. "
            f"Nearest police station is {police['name']}, "
            f"{police['distance_text']}. "
            f"Please dispatch emergency services immediately. "
            f"This is an automated message from the traffic monitoring system."
            "</Say>"
            "</Response>"
        )

        call = client.calls.create(
            twiml=twiml,
            from_=TWILIO_FROM_NUMBER,
            to=TWILIO_CALL_TO,
        )
        logger.info("Auto-call placed to %s | SID: %s", TWILIO_CALL_TO, call.sid)
        return True

    except Exception as e:
        logger.error("Auto-call failed: %s", e)
        return False


def send_control_room_email(accident_type, confidence, timestamp_s, hospital, police,
                              location, maps_link, time_str, screenshot_path=None, call_made=False):
    try:
        urgency = "CRITICAL" if confidence > 0.85 else "MODERATE"
        urgency_color = "#c0392b" if urgency == "CRITICAL" else "#e67e22"

        msg = MIMEMultipart()
        msg["Subject"] = f"[{urgency}] ROAD ACCIDENT - {accident_type} - Auto-Dispatch Alert"
        msg["From"] = EMAIL_SENDER
        msg["To"] = CONTROL_ROOM_EMAIL

        hospital_dist = hospital.get("distance_text", "Unknown distance")
        police_dist = police.get("distance_text", "Unknown distance")

        call_status_html = ""
        if AUTO_CALL_ENABLED:
            call_status_html = f"""
            <div style="background:{'#d4edda' if call_made else '#f8d7da'};padding:12px;border-radius:6px;margin:10px 0;">
              <b>{'✅ Automated voice call placed' if call_made else '⚠️ Automated voice call failed'}</b>
              to verified number for this alert.
            </div>
            """

        body = f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">

          <div style="background:{urgency_color};color:white;padding:20px;border-radius:8px;">
            <h2 style="margin:0;">🚨 TRAFFIC ACCIDENT DETECTED — {urgency}</h2>
            <p style="margin:5px 0 0;">Automated 24/7 CCTV Monitoring System</p>
          </div>

          <div style="background:white;padding:20px;margin-top:10px;border-radius:8px;">
            {call_status_html}

            <p><b>Action required:</b> Dispatch emergency services to the location below.</p>

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

    call_ok = False
    if AUTO_CALL_ENABLED:
        call_ok = make_auto_call(accident_type, confidence, location, hospital, police)
        print(f"  Auto-Call : {'OK' if call_ok else 'FAILED'}")
    else:
        print("  Auto-Call : Disabled (Twilio not configured in secrets)")

    email_ok = send_control_room_email(
        accident_type, confidence, timestamp_s,
        hospital, police, location, maps_link, time_str, screenshot_path,
        call_made=call_ok,
    )
    print(f"  Email     : {'OK' if email_ok else 'FAILED'}")
    print("=" * 55)

    return {
        "hospital": hospital,
        "police": police,
        "location": location,
        "ntfy": ntfy_ok,
        "call": call_ok,
        "email": email_ok,
    }
