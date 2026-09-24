"""Fetch the latest LTA camera locations and one image per available camera."""

from __future__ import annotations

import json
import os
import csv
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
ENDPOINT = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
OUTPUT_DIR = PROJECT_ROOT / "results" / "00_lta_traffic_images"
IMAGE_DIR = OUTPUT_DIR / "images"


def load_dotenv(path: Path) -> None:
    """读取简单的 KEY=VALUE 配置，不输出密钥内容。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fetch_cameras(api_key: str) -> list[dict]:
    """请求最新摄像头列表。"""
    request = Request(ENDPOINT, headers={"AccountKey": api_key, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"LTA API returned HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach LTA API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("LTA API response was not valid JSON") from exc
    cameras = payload.get("value", [])
    if not isinstance(cameras, list) or not cameras:
        raise RuntimeError("LTA API returned no camera records")
    return cameras


def download_image(camera: dict) -> tuple[str | None, str]:
    """下载单个摄像头图片并使用临时文件避免半成品。"""
    camera_id = str(camera.get("CameraID", "unknown"))
    image_link = camera.get("ImageLink")
    if not image_link:
        return None, "missing_image_link"
    destination = IMAGE_DIR / f"camera_{camera_id}.jpg"
    temporary = destination.with_suffix(".jpg.download")
    try:
        with urlopen(Request(image_link, method="GET"), timeout=30) as response:
            temporary.write_bytes(response.read())
        temporary.replace(destination)
        return str(destination.resolve()), "downloaded"
    except (HTTPError, URLError, OSError) as exc:
        if temporary.exists():
            temporary.unlink()
        return None, f"download_failed: {exc}"


def write_camera_map(locations: list[dict]) -> Path:
    """生成摄像头交互地图，并为每个标记提供 Google Maps 链接。"""
    valid_locations = [
        item for item in locations
        if isinstance(item.get("Latitude"), (int, float))
        and isinstance(item.get("Longitude"), (int, float))
    ]
    if not valid_locations:
        raise RuntimeError("No camera coordinates available for map generation")

    marker_data = []
    for item in valid_locations:
        camera_id = str(item.get("CameraID", "unknown"))
        latitude = float(item["Latitude"])
        longitude = float(item["Longitude"])
        google_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
        marker_data.append({
            "camera_id": camera_id,
            "latitude": latitude,
            "longitude": longitude,
            "google_url": google_url,
            "image_status": item.get("ImageDownloadStatus", "unknown"),
        })

    # 使用内嵌 SVG 绘制离线位置图，避免外部地图瓦片或 CDN 返回 403。
    marker_json = json.dumps(marker_data, ensure_ascii=False)
    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LTA Traffic Camera Locations</title>
  <style>body {{ font-family: Arial, sans-serif; margin: 20px; }} svg {{ width: 100%; max-width: 1200px; height: auto; border: 1px solid #bbb; background: #eef4f8; }} .camera {{ fill: #d22; stroke: #fff; stroke-width: 2; cursor: pointer; }} text {{ font-size: 14px; }} table {{ border-collapse: collapse; margin-top: 16px; }} td, th {{ border: 1px solid #ccc; padding: 6px 10px; }}</style>
</head>
<body>
<h2>LTA Traffic Camera Locations</h2>
<p>This offline map contains the latest coordinates. Click a marker or Google Maps link to open the exact location.</p>
<svg id="map" viewBox="0 0 1200 700" role="img" aria-label="Camera locations"></svg>
<table><thead><tr><th>Camera ID</th><th>Latitude</th><th>Longitude</th><th>Google Maps</th></tr></thead><tbody id="camera-table"></tbody></table>
<script>
const cameras = {marker_json};
const svg = document.getElementById('map');
const minLat = Math.min(...cameras.map(c => c.latitude)), maxLat = Math.max(...cameras.map(c => c.latitude));
const minLon = Math.min(...cameras.map(c => c.longitude)), maxLon = Math.max(...cameras.map(c => c.longitude));
const latRange = Math.max(maxLat - minLat, 0.001), lonRange = Math.max(maxLon - minLon, 0.001);
cameras.forEach(camera => {{
  const x = 60 + ((camera.longitude - minLon) / lonRange) * 1080;
  const y = 640 - ((camera.latitude - minLat) / latRange) * 580;
  const marker = document.createElementNS('http://www.w3.org/2000/svg', 'a');
  marker.setAttribute('href', camera.google_url); marker.setAttribute('target', '_blank'); marker.setAttribute('rel', 'noopener');
  marker.innerHTML = '<circle class="camera" cx="' + x + '" cy="' + y + '" r="9"><title>Camera ' + camera.camera_id + '</title></circle><text x="' + (x + 12) + '" y="' + (y - 12) + '">' + camera.camera_id + '</text>';
  svg.appendChild(marker);
  document.getElementById('camera-table').insertAdjacentHTML('beforeend', '<tr><td>' + camera.camera_id + '</td><td>' + camera.latitude.toFixed(6) + '</td><td>' + camera.longitude.toFixed(6) + '</td><td><a href="' + camera.google_url + '" target="_blank" rel="noopener">Open in Google Maps</a></td></tr>');
}});
</script>
</body>
</html>
'''
    output_path = OUTPUT_DIR / "camera_locations_map.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    load_dotenv(ENV_FILE)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError(f"API_KEY was not found in {ENV_FILE}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    locations = []
    for camera in fetch_cameras(api_key):
        location = {
            "CameraID": camera.get("CameraID"),
            "Latitude": camera.get("Latitude"),
            "Longitude": camera.get("Longitude"),
            "ImageLink": camera.get("ImageLink"),
        }
        local_image, status = download_image(location)
        location["LocalImage"] = local_image
        location["ImageDownloadStatus"] = status
        locations.append(location)
        print(f"Camera {location['CameraID']}: {status}")

    result = {
        "endpoint": ENDPOINT,
        "fetched_at": fetched_at,
        "camera_count": len(locations),
        "cameras": locations,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "camera_locations.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "image_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["camera_id", "image_path"])
        writer.writeheader()
        writer.writerows(
            {"camera_id": str(item["CameraID"]), "image_path": item["LocalImage"]}
            for item in locations
            if item.get("LocalImage")
        )
    map_path = write_camera_map(locations)
    print(f"Fetched {len(locations)} cameras")
    print(f"Saved camera map to: {map_path}")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
