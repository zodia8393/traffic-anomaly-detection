"""6대 동시 HLS 직접 다운로드 테스트 (3분)."""
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import requests

DURATION = 180  # 3분
OUTPUT = Path("/DATA/cctv_recording/20260528/test_6cam")
OUTPUT.mkdir(parents=True, exist_ok=True)

with open("/tmp/test_6cam.json") as f:
    cameras = json.load(f)


def resolve_media(hls_url):
    result = subprocess.run(
        ["ffprobe", "-v", "verbose", "-i", hls_url,
         "-show_entries", "format=duration", "-of", "csv=p=0"],
        capture_output=True, text=True, timeout=15,
    )
    for line in result.stderr.split("\n"):
        if "main_stream.m3u8" in line and "Opening" in line:
            match = re.search(r"'(http[^']+main_stream\.m3u8[^']*)'", line)
            if match:
                url = match.group(1)
                return url, url.rsplit("/", 1)[0] + "/"
    return None, None


def record_camera(idx, name, hls_url):
    slug = name.replace("[", "").replace("]", "").replace(" ", "_")
    out_file = OUTPUT / f"{slug}.ts"
    session = requests.Session()
    downloaded = set()
    total_bytes = 0
    total_segs = 0
    session_count = 0
    start = time.time()

    while time.time() - start < DURATION:
        session_count += 1
        media_url, base_url = resolve_media(hls_url)
        if not media_url:
            print(f"  [{idx}] {slug}: media 해석 실패, 3초 후 재시도")
            time.sleep(3)
            continue

        consecutive_empty = 0
        with open(out_file, "ab") as f:
            while time.time() - start < DURATION:
                try:
                    resp = session.get(media_url, timeout=10)
                    if resp.status_code != 200:
                        break

                    for line in resp.text.strip().split("\n"):
                        line = line.strip()
                        if line.startswith("#") or not line:
                            continue
                        seg_key = line.split("?")[0]
                        if seg_key in downloaded:
                            continue
                        seg_url = base_url + line
                        try:
                            sr = session.get(seg_url, timeout=10)
                            if sr.status_code == 200:
                                f.write(sr.content)
                                downloaded.add(seg_key)
                                total_bytes += len(sr.content)
                                total_segs += 1
                                consecutive_empty = 0
                        except Exception:
                            pass

                    if not any(l.split("?")[0] not in downloaded
                              for l in resp.text.strip().split("\n")
                              if not l.startswith("#") and l.strip()):
                        consecutive_empty += 1

                    if consecutive_empty > 10:
                        break

                    time.sleep(1.5)
                except Exception:
                    break
        time.sleep(2)

    elapsed = time.time() - start
    size_mb = total_bytes / (1024**2)
    print(f"  [{idx}] {slug}: {total_segs}seg, {size_mb:.1f}MB, {session_count}세션, {elapsed:.0f}초")


print(f"=== 6대 동시 HLS 직접 다운로드 테스트 ({DURATION}초) ===")
threads = []
for i, cam in enumerate(cameras):
    t = threading.Thread(target=record_camera, args=(i+1, cam["name"], cam["url"]))
    threads.append(t)

t0 = time.time()
for t in threads:
    t.start()
    time.sleep(0.3)

for t in threads:
    t.join()

total_files = list(OUTPUT.glob("*.ts"))
total_size = sum(f.stat().st_size for f in total_files) / (1024**2)
print(f"\n=== 결과: {len(total_files)}대 완료, 총 {total_size:.1f}MB, {time.time()-t0:.0f}초 ===")
