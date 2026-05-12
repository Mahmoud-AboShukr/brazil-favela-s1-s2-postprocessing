#!/usr/bin/env python3
"""
Download Sentinel-1 SAFE / COG_SAFE products from Copernicus Data Space Ecosystem (CDSE).

This is a corrected version of 07c.

Main fix
--------
The previous version searched CDSE using only exact product names such as:

    Name eq 'S1A_IW_GRDH_1SDV_20220413T082249_20220413T082318_042752_051A1B.SAFE'

That failed because CDSE may store Sentinel-1 products with an extra unique suffix:

    S1A_IW_GRDH_1SDV_20220413T082249_20220413T082318_042752_051A1B_ABCD.SAFE

or as COG_SAFE products:

    S1A_IW_GRDH_1SDV_20220413T082249_20220413T082318_042752_051A1B_ABCD_COG.SAFE

This script now searches using:
    1. Exact name
    2. Prefix search using startswith(Name, base_product_id)
    3. COG prefix search
    4. Time-window fallback search around the acquisition start time

It also writes a candidate-search CSV so we can see exactly what CDSE returned.

Authentication
--------------
Set credentials in PowerShell:

    $env:CDSE_USERNAME="your_username_or_email"
    $env:CDSE_PASSWORD="your_password"

If your account uses 2FA:

    $env:CDSE_TOTP="your_current_2fa_code"

Examples
--------
Dry run pilot products:

    python src/favela_postprocessing/07c_download_s1_safe_from_cdse.py --config configs/default.yaml --pilot-only --dry-run

Download Rio only:

    python src/favela_postprocessing/07c_download_s1_safe_from_cdse.py --config configs/default.yaml --city rio_de_janeiro

Download Rio only, forcing redownload:

    python src/favela_postprocessing/07c_download_s1_safe_from_cdse.py --config configs/default.yaml --city rio_de_janeiro --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


SCRIPT_NAME = "07c_download_s1_safe_from_cdse.py"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

CATALOGUE_ODATA_ROOT = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_ODATA_ROOT = "https://download.dataspace.copernicus.eu/odata/v1"

PILOT_CITIES = [
    "rio_de_janeiro",
    "belem",
    "porto_alegre",
]

DEFAULT_CHUNK_SIZE_MB = 8
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 120

S1_NAME_RE = re.compile(
    r"^(?P<prefix>"
    r"(?P<platform>S1[A-Z])_"
    r"(?P<mode>[A-Z0-9]+)_"
    r"(?P<product_type>[A-Z0-9]+)_"
    r"(?P<resolution_class>[A-Z0-9])"
    r"(?P<processing_level>[A-Z0-9])"
    r"(?P<product_class>[A-Z0-9])"
    r"(?P<polarization>[A-Z0-9]+)_"
    r"(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})_"
    r"(?P<absolute_orbit>\d+)_"
    r"(?P<datatake>[A-Fa-f0-9]+)"
    r")"
    r"(?:_(?P<unique_id>[A-Fa-f0-9]+))?"
    r"(?:_(?P<cog>COG))?"
    r"\.SAFE(?:\.zip)?$"
)

def is_downloadable_safe_candidate(product: Dict[str, Any]) -> bool:
    """
    Keep only catalogue products that look like downloadable Sentinel-1 SAFE products.

    This excludes auxiliary/non-SAFE products such as:
        *_CARD_BS
    while keeping:
        *.SAFE
        *_COG.SAFE
    """
    name = str(product.get("Name", "")).strip().upper()
    return name.endswith(".SAFE") or name.endswith(".SAFE.ZIP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 SAFE / COG_SAFE products from CDSE OData."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Manifest CSV path. Default is pilot manifest when --pilot-only is used, "
            "otherwise full manifest."
        ),
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Download only one city. Can be repeated.",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Use/download only pilot cities: rio_de_janeiro, belem, porto_alegre.",
    )
    parser.add_argument(
        "--endpoint",
        choices=["auto", "value", "zip"],
        default="auto",
        help=(
            "Download endpoint. auto tries /$value first, then /$zip. "
            "For older Sentinel-1 products, /$value is usually the safer default."
        ),
    )
    parser.add_argument(
        "--username-env",
        type=str,
        default="CDSE_USERNAME",
        help="Environment variable containing CDSE username. Default: CDSE_USERNAME",
    )
    parser.add_argument(
        "--password-env",
        type=str,
        default="CDSE_PASSWORD",
        help="Environment variable containing CDSE password. Default: CDSE_PASSWORD",
    )
    parser.add_argument(
        "--totp-env",
        type=str,
        default="CDSE_TOTP",
        help="Optional environment variable containing 2FA TOTP code. Default: CDSE_TOTP",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ZIP files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search products and show actions, but do not download.",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Limit number of products to process. Useful for testing.",
    )
    parser.add_argument(
        "--prefer-cog",
        action="store_true",
        help="Prefer COG_SAFE candidates when both original SAFE and COG_SAFE are found.",
    )
    parser.add_argument(
        "--allow-cog",
        action="store_true",
        default=True,
        help="Allow COG_SAFE fallback candidates. Default: true.",
    )
    parser.add_argument(
        "--no-cog",
        action="store_true",
        help="Disable COG_SAFE fallback candidates.",
    )
    parser.add_argument(
        "--time-window-minutes",
        type=int,
        default=10,
        help="Minutes around acquisition start for fallback time-window search. Default: 10",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Number of retries per request. Default: {DEFAULT_RETRIES}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=DEFAULT_CHUNK_SIZE_MB,
        help=f"Download chunk size in MB. Default: {DEFAULT_CHUNK_SIZE_MB}",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "output_root" not in cfg:
        raise KeyError("Missing required config key: output_root")

    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_city_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def clean_safe_name(value: str) -> str:
    name = str(value).strip()

    if not name:
        return ""

    if name.endswith(".SAFE.zip"):
        name = name[:-4]

    if not name.endswith(".SAFE"):
        name = f"{name}.SAFE"

    return name


def safe_base_prefix(safe_name: str) -> str:
    """
    Return the stable prefix before the optional final unique ID and before _COG.

    Example:
        S1A_IW_GRDH_1SDV_20220413T082249_20220413T082318_042752_051A1B.SAFE

    returns:
        S1A_IW_GRDH_1SDV_20220413T082249_20220413T082318_042752_051A1B
    """
    safe_name = clean_safe_name(safe_name)

    match = S1_NAME_RE.match(safe_name)

    if match:
        return match.group("prefix")

    return safe_name.replace(".SAFE", "").replace(".zip", "")


def parse_s1_name(safe_name: str) -> Dict[str, Any]:
    safe_name = clean_safe_name(safe_name)
    match = S1_NAME_RE.match(safe_name)

    if not match:
        return {
            "prefix": safe_base_prefix(safe_name),
            "start_datetime": None,
            "stop_datetime": None,
            "mode": "",
            "product_type": "",
            "polarization": "",
        }

    groups = match.groupdict()

    def parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

    return {
        "prefix": groups.get("prefix", safe_base_prefix(safe_name)),
        "start_datetime": parse_dt(groups.get("start")),
        "stop_datetime": parse_dt(groups.get("stop")),
        "mode": groups.get("mode", ""),
        "product_type": groups.get("product_type", ""),
        "polarization": groups.get("polarization", ""),
        "platform": groups.get("platform", ""),
        "absolute_orbit": groups.get("absolute_orbit", ""),
        "datatake": groups.get("datatake", ""),
    }


def dt_to_odata(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def odata_quote_string(value: str) -> str:
    return value.replace("'", "''")


def http_json_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url=url,
                data=data,
                headers=headers or {},
                method=method,
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read()
                text = body.decode("utf-8")
                return json.loads(text)

        except Exception as exc:
            last_error = exc

            if attempt < retries:
                wait_seconds = 2 * attempt
                print(f"[WARN] HTTP JSON request failed, retrying in {wait_seconds}s: {exc}")
                time.sleep(wait_seconds)

    raise RuntimeError(f"HTTP JSON request failed after {retries} attempts: {last_error}")


def get_access_token(
    username: str,
    password: str,
    totp: Optional[str],
    timeout: int,
    retries: int,
) -> str:
    payload = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    if totp:
        payload["totp"] = totp

    data = urllib.parse.urlencode(payload).encode("utf-8")

    result = http_json_request(
        url=TOKEN_URL,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=data,
        timeout=timeout,
        retries=retries,
    )

    token = result.get("access_token")

    if not token:
        error = result.get("error", "")
        description = result.get("error_description", "")
        raise RuntimeError(
            f"Could not obtain CDSE access token. error={error}, description={description}"
        )

    return str(token)


def make_products_query(filter_expr: str, top: int = 20) -> str:
    params = {
        "$filter": filter_expr,
        "$select": (
            "Id,Name,ContentLength,Online,S3Path,"
            "ContentDate,PublicationDate,ModificationDate"
        ),
        "$top": str(top),
    }

    return f"{CATALOGUE_ODATA_ROOT}/Products?{urllib.parse.urlencode(params)}"


def query_products(
    filter_expr: str,
    timeout: int,
    retries: int,
    top: int = 20,
) -> List[Dict[str, Any]]:
    url = make_products_query(filter_expr, top=top)

    result = http_json_request(
        url=url,
        method="GET",
        headers={"Accept": "application/json"},
        timeout=timeout,
        retries=retries,
    )

    values = result.get("value", [])

    if not isinstance(values, list):
        return []

    return values


def product_key(product: Dict[str, Any]) -> str:
    return str(product.get("Id", "")) or str(product.get("Name", ""))


def add_products_unique(
    out: List[Dict[str, Any]],
    products: List[Dict[str, Any]],
    search_method: str,
    search_filter: str,
) -> None:
    seen = {product_key(item) for item in out}

    for product in products:
        key = product_key(product)

        if not key or key in seen:
            continue

        product = dict(product)
        product["_search_method"] = search_method
        product["_search_filter"] = search_filter

        out.append(product)
        seen.add(key)


def search_product_candidates(
    safe_name: str,
    timeout: int,
    retries: int,
    allow_cog: bool = True,
    time_window_minutes: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search CDSE using several increasingly flexible queries.
    """
    safe_name = clean_safe_name(safe_name)
    prefix = safe_base_prefix(safe_name)
    parsed = parse_s1_name(safe_name)

    candidates: List[Dict[str, Any]] = []

    collection_filter = "Collection/Name eq 'SENTINEL-1'"

    exact_filter = (
        f"{collection_filter} and "
        f"Name eq '{odata_quote_string(safe_name)}'"
    )

    products = query_products(
        exact_filter,
        timeout=timeout,
        retries=retries,
        top=10,
    )
    add_products_unique(candidates, products, "exact_name", exact_filter)

    prefix_filter = (
        f"{collection_filter} and "
        f"startswith(Name,'{odata_quote_string(prefix)}')"
    )

    products = query_products(
        prefix_filter,
        timeout=timeout,
        retries=retries,
        top=20,
    )
    add_products_unique(candidates, products, "prefix_startswith", prefix_filter)

    if allow_cog:
        cog_filter = (
            f"{collection_filter} and "
            f"startswith(Name,'{odata_quote_string(prefix)}') and "
            f"contains(Name,'COG')"
        )

        products = query_products(
            cog_filter,
            timeout=timeout,
            retries=retries,
            top=20,
        )
        add_products_unique(candidates, products, "prefix_cog", cog_filter)

    start_dt = parsed.get("start_datetime")
    mode = parsed.get("mode", "")
    product_type = parsed.get("product_type", "")

    if isinstance(start_dt, datetime):
        dt_min = start_dt - timedelta(minutes=time_window_minutes)
        dt_max = start_dt + timedelta(minutes=time_window_minutes)

        time_filter = (
            f"{collection_filter} and "
            f"ContentDate/Start ge {dt_to_odata(dt_min)} and "
            f"ContentDate/Start le {dt_to_odata(dt_max)}"
        )

        if mode:
            time_filter += f" and contains(Name,'{odata_quote_string(mode)}')"

        if product_type:
            time_filter += f" and contains(Name,'{odata_quote_string(product_type)}')"

        products = query_products(
            time_filter,
            timeout=timeout,
            retries=retries,
            top=50,
        )
        add_products_unique(candidates, products, "time_window", time_filter)

        if allow_cog:
            time_cog_filter = time_filter + " and contains(Name,'COG')"

            products = query_products(
                time_cog_filter,
                timeout=timeout,
                retries=retries,
                top=50,
            )
            add_products_unique(candidates, products, "time_window_cog", time_cog_filter)

    return candidates


def is_cog_product_name(name: str) -> bool:
    upper = str(name).upper()
    return "COG" in upper or upper.endswith("_COG.SAFE")


def score_product_candidate(
    safe_name: str,
    product: Dict[str, Any],
    prefer_cog: bool,
) -> Tuple[int, str]:
    safe_name = clean_safe_name(safe_name)
    prefix = safe_base_prefix(safe_name)
    name = str(product.get("Name", ""))

    score = 0
    reasons: List[str] = []

    if name == safe_name:
        score += 1000
        reasons.append("exact_name")

    if name.startswith(prefix):
        score += 800
        reasons.append("startswith_prefix")

    if prefix in name:
        score += 500
        reasons.append("contains_prefix")

    parsed = parse_s1_name(safe_name)

    for key in ["platform", "mode", "product_type", "polarization", "absolute_orbit", "datatake"]:
        value = parsed.get(key)
        if value and str(value) in name:
            score += 25
            reasons.append(f"contains_{key}")

    if is_cog_product_name(name):
        if prefer_cog:
            score += 80
            reasons.append("prefer_cog")
        else:
            score -= 20
            reasons.append("cog_fallback")
    else:
        score += 50
        reasons.append("original_safe_preferred")

    search_method = str(product.get("_search_method", ""))

    if search_method == "exact_name":
        score += 100
    elif search_method == "prefix_startswith":
        score += 80
    elif search_method == "prefix_cog":
        score += 60
    elif search_method == "time_window":
        score += 30
    elif search_method == "time_window_cog":
        score += 20

    if not reasons:
        reasons.append("weak_match")

    return score, "+".join(reasons)


def choose_best_product(
    safe_name: str,
    candidates: List[Dict[str, Any]],
    prefer_cog: bool,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    scored: List[Dict[str, Any]] = []

    for product in candidates:
        if not is_downloadable_safe_candidate(product):
            continue

        score, reason = score_product_candidate(
            safe_name=safe_name,
            product=product,
            prefer_cog=prefer_cog,
        )

        item = dict(product)
        item["_candidate_score"] = score
        item["_candidate_reason"] = reason
        scored.append(item)

    scored = sorted(
        scored,
        key=lambda item: int(item.get("_candidate_score", 0)),
        reverse=True,
    )

    if not scored:
        return None, []

    return scored[0], scored


def human_bytes(num_bytes: Any) -> str:
    if num_bytes is None or num_bytes == "":
        return ""

    try:
        value = float(num_bytes)
    except Exception:
        return ""

    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PB"


def is_valid_safe_zip(path: Path) -> Tuple[bool, str]:
    if not path.exists() or not path.is_file():
        return False, "file_missing"

    try:
        if not zipfile.is_zipfile(path):
            return False, "not_a_zip_file"

        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()

            if not names:
                return False, "empty_zip"

            normalized = [name.replace("\\", "/").lower() for name in names]

            has_manifest = any(name.endswith("manifest.safe") for name in normalized)
            has_measurement = any("/measurement/" in name for name in normalized)
            has_annotation = any("/annotation/" in name for name in normalized)

            if not has_manifest:
                return False, "missing_manifest_safe"

            if not has_measurement:
                return False, "missing_measurement_folder"

            if not has_annotation:
                return False, "missing_annotation_folder"

            return True, "valid_safe_zip"

    except Exception as exc:
        return False, repr(exc)


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def endpoint_url(product_id: str, endpoint: str) -> str:
    if endpoint == "value":
        return f"{DOWNLOAD_ODATA_ROOT}/Products({product_id})/$value"

    if endpoint == "zip":
        return f"{DOWNLOAD_ODATA_ROOT}/Products({product_id})/$zip"

    raise ValueError(f"Unsupported endpoint: {endpoint}")


def download_url_to_file(
    url: str,
    output_path: Path,
    token: str,
    timeout: int,
    retries: int,
    chunk_size: int,
) -> Tuple[bool, str, int, Optional[int]]:
    ensure_dir(output_path.parent)

    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    if temp_path.exists():
        temp_path.unlink(missing_ok=True)

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "favela-dataset-s1-safe-downloader/2.0",
    }

    last_error = ""

    for attempt in range(1, retries + 1):
        bytes_written = 0
        expected_size: Optional[int] = None

        try:
            req = urllib.request.Request(
                url=url,
                headers=headers,
                method="GET",
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", None)

                if status is not None and status >= 400:
                    raise RuntimeError(f"HTTP status {status}")

                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    expected_size = int(content_length)

                total_for_tqdm = expected_size if expected_size and expected_size > 0 else None

                with temp_path.open("wb") as f:
                    with tqdm(
                        total=total_for_tqdm,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {output_path.name}",
                    ) as progress:
                        while True:
                            chunk = response.read(chunk_size)

                            if not chunk:
                                break

                            f.write(chunk)
                            bytes_written += len(chunk)
                            progress.update(len(chunk))

            if expected_size is not None and bytes_written != expected_size:
                raise RuntimeError(
                    f"Incomplete download: wrote {bytes_written}, expected {expected_size}"
                )

            temp_path.replace(output_path)
            return True, "OK", bytes_written, expected_size

        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""

            last_error = f"HTTPError {exc.code}: {error_body[:800]}"

        except Exception as exc:
            last_error = repr(exc)

        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        if attempt < retries:
            wait_seconds = 5 * attempt
            print(f"[WARN] Download attempt {attempt} failed: {last_error}")
            print(f"[WARN] Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    return False, last_error, 0, None


def download_product(
    product_id: str,
    output_path: Path,
    token: str,
    endpoint_mode: str,
    timeout: int,
    retries: int,
    chunk_size: int,
) -> Dict[str, Any]:
    if endpoint_mode == "auto":
        endpoints = ["value", "zip"]
    else:
        endpoints = [endpoint_mode]

    errors: List[str] = []

    for endpoint in endpoints:
        url = endpoint_url(product_id, endpoint)
        print(f"[INFO] Trying endpoint /${endpoint}: {url}")

        ok, message, bytes_written, expected_size = download_url_to_file(
            url=url,
            output_path=output_path,
            token=token,
            timeout=timeout,
            retries=retries,
            chunk_size=chunk_size,
        )

        if not ok:
            errors.append(f"{endpoint}: {message}")
            continue

        zip_valid, zip_reason = is_valid_safe_zip(output_path)

        if zip_valid:
            return {
                "download_ok": True,
                "download_endpoint_used": endpoint,
                "download_message": message,
                "bytes_written": bytes_written,
                "expected_size": expected_size,
                "zip_valid": True,
                "zip_validation_reason": zip_reason,
            }

        errors.append(f"{endpoint}: downloaded file failed SAFE ZIP validation: {zip_reason}")

        if output_path.exists():
            output_path.unlink(missing_ok=True)

    return {
        "download_ok": False,
        "download_endpoint_used": "",
        "download_message": " | ".join(errors),
        "bytes_written": 0,
        "expected_size": None,
        "zip_valid": False,
        "zip_validation_reason": "all_endpoints_failed",
    }


def load_manifest(
    output_root: Path,
    manifest_path: Optional[Path],
    pilot_only: bool,
) -> pd.DataFrame:
    if manifest_path is None:
        if pilot_only:
            manifest_path = output_root / "metadata" / "s1_safe_download_manifest_pilot.csv"
        else:
            manifest_path = output_root / "metadata" / "s1_safe_download_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = ["city", "safe_name", "target_safe_zip"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Manifest is missing required columns: {missing}")

    df["city"] = df["city"].map(normalize_city_name)
    df["safe_name"] = df["safe_name"].map(clean_safe_name)

    return df


def filter_manifest(
    df: pd.DataFrame,
    selected_cities: Optional[Sequence[str]],
    pilot_only: bool,
    max_products: Optional[int],
) -> pd.DataFrame:
    out = df.copy()

    if pilot_only:
        out = out[out["city"].isin(PILOT_CITIES)]

    if selected_cities:
        cities = {normalize_city_name(city) for city in selected_cities}
        out = out[out["city"].isin(cities)]

    out = out.sort_values(["city", "safe_name"]).reset_index(drop=True)

    if max_products is not None:
        out = out.head(max_products)

    return out


def target_zip_from_row(row: pd.Series) -> Path:
    value = str(row.get("target_safe_zip", "")).strip()

    if not value:
        raise ValueError(f"Manifest row for city={row.get('city')} has no target_safe_zip.")

    return Path(value)


def process_row(
    row: pd.Series,
    token: Optional[str],
    endpoint_mode: str,
    overwrite: bool,
    dry_run: bool,
    timeout: int,
    retries: int,
    chunk_size: int,
    allow_cog: bool,
    prefer_cog: bool,
    time_window_minutes: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    city = normalize_city_name(str(row["city"]))
    safe_name = clean_safe_name(str(row["safe_name"]))
    target_zip = target_zip_from_row(row)

    base: Dict[str, Any] = {
        "city": city,
        "requested_safe_name": safe_name,
        "target_safe_zip": str(target_zip),
        "status": "",
        "selected_product_id": "",
        "selected_product_name": "",
        "selected_product_is_cog": False,
        "selected_search_method": "",
        "selected_candidate_score": "",
        "selected_candidate_reason": "",
        "candidate_count": 0,
        "catalogue_online": "",
        "catalogue_content_length": "",
        "catalogue_content_length_human": "",
        "download_endpoint_used": "",
        "bytes_written": "",
        "expected_size": "",
        "zip_valid": False,
        "zip_validation_reason": "",
        "message": "",
    }

    if target_zip.exists() and not overwrite:
        valid, reason = is_valid_safe_zip(target_zip)

        base.update(
            {
                "status": "SKIPPED_EXISTS_VALID" if valid else "SKIPPED_EXISTS_INVALID",
                "bytes_written": file_size(target_zip),
                "zip_valid": valid,
                "zip_validation_reason": reason,
                "message": (
                    "Existing ZIP is valid; use --overwrite to redownload."
                    if valid
                    else "Existing file is not a valid SAFE ZIP; use --overwrite to redownload."
                ),
            }
        )

        return base, []

    candidates = search_product_candidates(
        safe_name=safe_name,
        timeout=timeout,
        retries=retries,
        allow_cog=allow_cog,
        time_window_minutes=time_window_minutes,
    )

    selected, scored_candidates = choose_best_product(
        safe_name=safe_name,
        candidates=candidates,
        prefer_cog=prefer_cog,
    )

    candidate_rows: List[Dict[str, Any]] = []

    for candidate in scored_candidates:
        candidate_rows.append(
            {
                "city": city,
                "requested_safe_name": safe_name,
                "candidate_product_id": candidate.get("Id", ""),
                "candidate_name": candidate.get("Name", ""),
                "candidate_is_cog": is_cog_product_name(str(candidate.get("Name", ""))),
                "candidate_score": candidate.get("_candidate_score", ""),
                "candidate_reason": candidate.get("_candidate_reason", ""),
                "search_method": candidate.get("_search_method", ""),
                "search_filter": candidate.get("_search_filter", ""),
                "content_length": candidate.get("ContentLength", ""),
                "content_length_human": human_bytes(candidate.get("ContentLength", "")),
                "online": candidate.get("Online", ""),
                "s3_path": candidate.get("S3Path", ""),
                "content_date": json.dumps(candidate.get("ContentDate", ""), ensure_ascii=False),
            }
        )

    base["candidate_count"] = len(scored_candidates)

    if selected is None:
        base.update(
            {
                "status": "FAILED_PRODUCT_NOT_FOUND",
                "message": (
                    "No CDSE product found using exact, prefix, COG, or time-window searches. "
                    "Check the requested product name in Copernicus Browser."
                ),
            }
        )
        return base, candidate_rows

    product_id = str(selected.get("Id", ""))
    selected_name = str(selected.get("Name", ""))

    base.update(
        {
            "selected_product_id": product_id,
            "selected_product_name": selected_name,
            "selected_product_is_cog": is_cog_product_name(selected_name),
            "selected_search_method": selected.get("_search_method", ""),
            "selected_candidate_score": selected.get("_candidate_score", ""),
            "selected_candidate_reason": selected.get("_candidate_reason", ""),
            "catalogue_online": selected.get("Online", ""),
            "catalogue_content_length": selected.get("ContentLength", ""),
            "catalogue_content_length_human": human_bytes(selected.get("ContentLength", "")),
        }
    )

    if not product_id:
        base.update(
            {
                "status": "FAILED_MISSING_PRODUCT_ID",
                "message": "CDSE product found but has no Id.",
            }
        )
        return base, candidate_rows

    if dry_run:
        base.update(
            {
                "status": "DRY_RUN_FOUND",
                "message": (
                    f"Product found. Selected catalogue name: {selected_name}. "
                    "Dry run did not download."
                ),
            }
        )
        return base, candidate_rows

    if token is None:
        base.update(
            {
                "status": "FAILED_NO_TOKEN",
                "message": "No access token available.",
            }
        )
        return base, candidate_rows

    result = download_product(
        product_id=product_id,
        output_path=target_zip,
        token=token,
        endpoint_mode=endpoint_mode,
        timeout=timeout,
        retries=retries,
        chunk_size=chunk_size,
    )

    if result["download_ok"]:
        base.update(
            {
                "status": "DOWNLOADED_OK",
                "download_endpoint_used": result["download_endpoint_used"],
                "bytes_written": result["bytes_written"],
                "expected_size": result["expected_size"],
                "zip_valid": result["zip_valid"],
                "zip_validation_reason": result["zip_validation_reason"],
                "message": result["download_message"],
            }
        )
    else:
        base.update(
            {
                "status": "FAILED_DOWNLOAD",
                "download_endpoint_used": result["download_endpoint_used"],
                "bytes_written": result["bytes_written"],
                "expected_size": result["expected_size"],
                "zip_valid": result["zip_valid"],
                "zip_validation_reason": result["zip_validation_reason"],
                "message": result["download_message"],
            }
        )

    return base, candidate_rows


def write_qc_outputs(
    output_root: Path,
    rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    qc_root = output_root / "qc"
    ensure_dir(qc_root)

    qc_path = qc_root / "s1_safe_cdse_download_qc.csv"
    candidates_path = qc_root / "s1_safe_cdse_search_candidates.csv"

    df = pd.DataFrame(rows)
    df.to_csv(qc_path, index=False)

    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df.to_csv(candidates_path, index=False)

    print(f"[INFO] Wrote QC CSV: {qc_path}")
    print(f"[INFO] Wrote candidate search CSV: {candidates_path}")

    if "status" in df.columns and not df.empty:
        print("[INFO] Status counts:")
        print(df["status"].value_counts(dropna=False).to_string())

    if not candidates_df.empty:
        print("[INFO] Candidate counts by city:")
        print(candidates_df.groupby("city")["candidate_name"].count().to_string())

    return qc_path, candidates_path


def main() -> int:
    args = parse_args()

    cfg = load_config(args.config)
    output_root = Path(str(cfg["output_root"]))

    allow_cog = bool(args.allow_cog) and not bool(args.no_cog)

    manifest = load_manifest(
        output_root=output_root,
        manifest_path=args.manifest,
        pilot_only=args.pilot_only,
    )

    manifest = filter_manifest(
        manifest,
        selected_cities=args.city,
        pilot_only=args.pilot_only,
        max_products=args.max_products,
    )

    print("[INFO] Sentinel-1 SAFE / COG_SAFE CDSE downloader")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Products selected: {len(manifest)}")
    print(f"[INFO] Endpoint mode: {args.endpoint}")
    print(f"[INFO] Dry run: {args.dry_run}")
    print(f"[INFO] Overwrite: {args.overwrite}")
    print(f"[INFO] Allow COG_SAFE fallback: {allow_cog}")
    print(f"[INFO] Prefer COG_SAFE: {args.prefer_cog}")
    print(f"[INFO] Time-window fallback: +/- {args.time_window_minutes} minutes")

    if manifest.empty:
        print("[WARN] No products selected from manifest.")
        write_qc_outputs(output_root, [], [])
        return 0

    username = os.environ.get(args.username_env)
    password = os.environ.get(args.password_env)
    totp = os.environ.get(args.totp_env)

    token: Optional[str] = None

    if args.dry_run:
        print("[INFO] Dry run: no token needed for product search.")
    else:
        if not username or not password:
            raise RuntimeError(
                f"Missing CDSE credentials. Set PowerShell environment variables:\n"
                f'  $env:{args.username_env}="your_username_or_email"\n'
                f'  $env:{args.password_env}="your_password"\n'
                f"If using 2FA, also set:\n"
                f'  $env:{args.totp_env}="your_current_2fa_code"'
            )

        print("[INFO] CDSE credentials found. A fresh token will be requested before each product.")

    chunk_size = max(1, args.chunk_size_mb) * 1024 * 1024

    rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="CDSE SAFE products"):
        city = normalize_city_name(str(row["city"]))
        safe_name = clean_safe_name(str(row["safe_name"]))

        print(f"[INFO] Processing {city}: {safe_name}")

        if not args.dry_run:
            print("[INFO] Requesting fresh CDSE access token for this product...")
            token = get_access_token(
                username=username,
                password=password,
                totp=totp,
                timeout=args.timeout,
                retries=args.retries,
            )
            print("[INFO] Fresh access token acquired.")

        result, candidates = process_row(
            row=row,
            token=token,
            endpoint_mode=args.endpoint,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            timeout=args.timeout,
            retries=args.retries,
            chunk_size=chunk_size,
            allow_cog=allow_cog,
            prefer_cog=args.prefer_cog,
            time_window_minutes=args.time_window_minutes,
        )

        rows.append(result)
        candidate_rows.extend(candidates)

        print(
            f"[INFO] {city}: {result['status']} - "
            f"candidates={result.get('candidate_count', 0)} - "
            f"selected={result.get('selected_product_name', '')}"
        )

        if candidates:
            print("[INFO] Top candidates:")
            for candidate in candidates[:5]:
                print(
                    "       "
                    f"score={candidate.get('candidate_score')} "
                    f"method={candidate.get('search_method')} "
                    f"name={candidate.get('candidate_name')}"
                )

    qc_path, candidates_path = write_qc_outputs(output_root, rows, candidate_rows)

    statuses = {str(row.get("status")) for row in rows}

    if any(status.startswith("FAILED") for status in statuses):
        print("[ERROR] Some products failed. Check:")
        print(f"        {qc_path}")
        print(f"        {candidates_path}")
        return 1

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())