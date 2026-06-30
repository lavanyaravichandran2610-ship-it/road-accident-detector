"""
Auto Camera Location Detection
Automatically detects the location of each camera using:
1. IP-based geolocation (works on any internet-connected camera)
2. Reverse geocoding to get full address
3. Nearest hospital and police station detection
No manual GPS entry needed!
"""

import requests
import logging
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache file to avoid repeated API calls
CACHE_FILE = Path("/tmp/camera_locations.json")
CACHE_TTL  = 3600  # refresh every 1 hour


# ==============================================================================
# 1. GET LOCATION FROM IP ADDRESS (Automatic)
# ==============================================================================

def get_camera_location(camera_ip: str = None):
    """
    Automatically detects camera location.

    How it works:
    - If camera_ip is given → gets location of that specific camera
    - If no IP given → gets location of current server/machine
    - Uses multiple free APIs as fallback

    Args:
        camera_ip: IP address of the camera (e.g. "192.168.1.100")
                   Leave None to use current machine's IP

    Returns:
        dict with lat, lng, city, region, address, country
    """

    # Check cache first
    cached = _load_cache(camera_ip or "self")
    if cached:
        logger.info("Using cached location for %s", camera_ip or "self")
        return cached

    location = None

    # Try multiple free IP geolocation APIs
    apis = [
        f"https://ipapi.co/{camera_ip + '/' if camera_ip else ''}json/",
        f"https://ip-api.com/json/{camera_ip or ''}",
        f"https://ipwho.is/{camera_ip or ''}",
    ]

    for api_url in apis:
        try:
            r = requests.get(api_url, timeout=8)
            if r.status_code == 200:
                data = r.json()

                # Handle different API response formats
                lat = (data.get("latitude") or
                       data.get("lat") or
                       data.get("location", {}).get("lat"))
                lng = (data.get("longitude") or
                       data.get("lon") or
                       data.get("location", {}).get("lng"))
                city = (data.get("city") or
                        data.get("city", "Unknown"))
                region = (data.get("region") or
                          data.get("regionName") or
                          data.get("region_name", "Unknown"))
                country = (data.get("country_name") or
                           data.get("country") or "India")

                if lat and lng:
                    location = {
                        "lat":     float(lat),
                        "lng":     float(lng),
                        "city":    city,
                        "region":  region,
                        "country": country,
                        "source":  api_url.split("/")[2],
                    }
                    break

        except Exception as e:
            logger.warning("API %s failed: %s", api_url, e)
            continue

    if not location:
        logger.error("All location APIs failed")
        return None

    # Get full address via reverse geocoding
    address = reverse_geocode(location["lat"], location["lng"])
    location["address"] = address

    # Save to cache
    _save_cache(camera_ip or "self", location)

    logger.info(
        "Camera location detected: %s, %s (%s, %s)",
        location["city"], location["region"],
        location["lat"], location["lng"],
    )
    return location


# ==============================================================================
# 2. REVERSE GEOCODE — Get address from GPS
# ==============================================================================

def reverse_geocode(lat, lng):
    """Convert GPS coordinates to a human-readable address."""
    try:
        headers = {"User-Agent": "RoadAccidentDetectionSystem/1.0"}
        r = requests.get(
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lng}&format=json",
            headers=headers,
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            addr = data.get("address", {})
            parts = [
                addr.get("road") or addr.get("neighbourhood"),
                addr.get("suburb") or addr.get("village"),
                addr.get("city") or addr.get("town"),
                addr.get("state"),
                addr.get("country"),
            ]
            return ", ".join(p for p in parts if p)
    except Exception as e:
        logger.warning("Reverse geocode failed: %s", e)

    return f"Location: {lat:.4f}, {lng:.4f}"


# ==============================================================================
# 3. FIND NEAREST HOSPITAL AND POLICE (Auto)
# ==============================================================================

def find_nearest_places(lat, lng):
    """
    Automatically finds nearest hospital and police station
    using OpenStreetMap — no API key needed.
    Returns both with distance and navigation links.
    """
    import math

    def distance_km(lat1, lng1, lat2, lng2):
        R = 6371
        d_lat = math.radians(lat2 - lat1)
        d_lng = math.radians(lng2 - lng1)
        a = (math.sin(d_lat/2)**2 +
             math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) *
             math.sin(d_lng/2)**2)
        return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)

    def search_place(place_type):
        default_phone = "108" if place_type == "hospital" else "100"
        default = {
            "name":     f"Nearest {place_type.title()}",
            "address":  "See map link",
            "phone":    default_phone,
            "distance": "Unknown",
            "nav_link": f"https://www.google.com/maps/search/{place_type}/@{lat},{lng},15z",
            "dir_link": f"https://www.google.com/maps/dir/{lat},{lng}/{place_type}",
        }
        try:
            headers = {"User-Agent": "RoadAccidentDetectionSystem/1.0"}
            r = requests.get(
                f"https://nominatim.openstreetmap.org/search"
                f"?q={place_type}&format=json&limit=1"
                f"&lat={lat}&lon={lng}&countrycodes=in",
                headers=headers,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if data:
                    p     = data[0]
                    p_lat = float(p.get("lat", lat))
                    p_lng = float(p.get("lon", lng))
                    dist  = distance_km(lat, lng, p_lat, p_lng)
                    name  = p.get("display_name", "").split(",")[0]
                    addr  = ", ".join(p.get("display_name", "").split(",")[:3])
                    default.update({
                        "name":     name,
                        "address":  addr,
                        "distance": f"{dist} km",
                        "nav_link": f"https://www.google.com/maps/search/{place_type}/@{p_lat},{p_lng},17z",
                        "dir_link": f"https://www.google.com/maps/dir/{lat},{lng}/{p_lat},{p_lng}",
                    })
        except Exception as e:
            logger.warning("Place search failed for %s: %s", place_type, e)
        return default

    hospital = search_place("hospital")
    police   = search_place("police")
    return hospital, police


# ==============================================================================
# 4. FULL AUTO CAMERA SETUP
# ==============================================================================

def auto_setup_camera(camera_id: str, camera_ip: str = None):
    """
    Fully automatic camera setup.
    Call this once when the camera connects.
    Returns complete camera info with location, hospital, police.

    Args:
        camera_id: unique ID like "CAM-001"
        camera_ip: IP of the camera (None = use server IP)
    """
    print(f"\nAuto-detecting location for {camera_id}...")

    location = get_camera_location(camera_ip)
    if not location:
        print(f"Could not detect location for {camera_id}")
        return None

    print(f"Location detected: {location['city']}, {location['region']}")
    print(f"GPS: {location['lat']}, {location['lng']}")
    print(f"Address: {location['address']}")

    print("Finding nearest hospital and police station...")
    hospital, police = find_nearest_places(location["lat"], location["lng"])

    print(f"Hospital: {hospital['name']} ({hospital['distance']})")
    print(f"Police:   {police['name']} ({police['distance']})")

    camera_info = {
        "camera_id": camera_id,
        "camera_ip": camera_ip or "auto",
        "location":  location,
        "hospital":  hospital,
        "police":    police,
        "maps_link": f"https://maps.google.com/?q={location['lat']},{location['lng']}",
        "detected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return camera_info


# ==============================================================================
# 5. CACHE HELPERS
# ==============================================================================

def _load_cache(key):
    try:
        if CACHE_FILE.exists():
            cache = json.loads(CACHE_FILE.read_text())
            entry = cache.get(key)
            if entry:
                age = time.time() - entry.get("cached_at", 0)
                if age < CACHE_TTL:
                    return entry.get("data")
    except Exception:
        pass
    return None


def _save_cache(key, data):
    try:
        cache = {}
        if CACHE_FILE.exists():
            cache = json.loads(CACHE_FILE.read_text())
        cache[key] = {"data": data, "cached_at": time.time()}
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass