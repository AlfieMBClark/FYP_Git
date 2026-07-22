import os
import re
import time
import argparse
import requests
import xml.etree.ElementTree as ET

WEBSITE_BASE = "http://aisdata.ais.dk/"
S3_BASE = "http://aisdata.ais.dk.s3.eu-central-1.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DEST_DIR = "downloads"

_DATE_IN_NAME = re.compile(r"aisdk-(\d{4})-\d{2}(?:-\d{2})?\.zip$")


def key_year(key: str) -> str:
    """Year a key belongs to, whether it is filed under a folder or not."""
    m = _DATE_IN_NAME.search(os.path.basename(key))
    if m:
        return m.group(1)
    return key.split("/")[0] if "/" in key else ""


def iter_zip_keys():
    marker = ""
    page = 0
    while True:
        params = {}
        if marker:
            params["marker"] = marker
        r = requests.get(S3_BASE, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys_found = 0
        for elem in root.iter(f"{NS}Key"):
            key = (elem.text or "").strip()
            if key.endswith(".zip"):
                yield key
            marker = key
            keys_found += 1
        page += 1
        is_truncated = root.findtext(f"{NS}IsTruncated", "false").lower()
        print(f"  Page {page}: {keys_found} keys, truncated={is_truncated}")
        if is_truncated != "true":
            break


def download(key, dest_dir=DEST_DIR, chunk_size=1024 * 1024):
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(key)
    # File by the year in the *filename*
    year = key_year(key)
    year_dir = os.path.join(dest_dir, year) if year else dest_dir
    os.makedirs(year_dir, exist_ok=True)
    out_path = os.path.join(year_dir, filename)

    url = WEBSITE_BASE + key

    if os.path.exists(out_path):
        local = os.path.getsize(out_path)
        try:
            remote = int(requests.head(url, timeout=30).headers.get("content-length", 0))
        except Exception:
            remote = 0
        if remote and local == remote:
            print(f"  Skipping (complete): {out_path}")
            return
        if remote:
            print(f"  Re-downloading (truncated: {local / 1048576:.0f} MB of "
                  f"{remote / 1048576:.0f} MB): {out_path}")
        else:
            print(f"  Skipping (exists, size unverified): {out_path}")
            return

    print(f"Downloading {key} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {downloaded / total * 100:.1f}%  ({downloaded // 1_048_576} MB / {total // 1_048_576} MB)\r", end="", flush=True)
    print(f"\n  -> saved to {out_path}")
    time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description="Download AIS zip files from aisdata.ais.dk")
    parser.add_argument(
        "years",
        nargs="*",
        metavar="YEAR",
        help="Years to download (e.g. 2024 2025). Omit to download everything.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to download after filtering.",
    )
    args = parser.parse_args()
    year_filter = set(args.years)

    print("Listing all .zip files from S3 bucket (paginated)...")
    all_keys = list(iter_zip_keys())

    if year_filter:
        keys = [k for k in all_keys if key_year(k) in year_filter]
        print(f"\nFiltered to years {sorted(year_filter)}: {len(keys)} of {len(all_keys)} zip files.\n")
    else:
        keys = all_keys
        print(f"\nFound {len(keys)} zip files. Starting downloads...\n")

    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit must be >= 0")
        keys = keys[: args.limit]
        print(f"Applying limit: {len(keys)} files will be downloaded.")

    if not keys:
        print("No matching .zip files found.")
        return

    for i, key in enumerate(keys, 1):
        print(f"[{i}/{len(keys)}]", end=" ")
        try:
            download(key)
        except Exception as e:
            print(f"\n  ERROR downloading {key}: {e}")


if __name__ == "__main__":
    main()
