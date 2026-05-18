#!/usr/bin/env python3
"""
Download selected fallback Sentinel-1 products from Copernicus Data Space OData.

Input:
    <instance-root>/qc/s1_fallback_selection/selected_s1_fallback_products.csv

For each selected item:
    selected_item_id -> selected_item_id.SAFE
    query Copernicus OData Products by exact Name
    download product ZIP from:
        https://download.dataspace.copernicus.eu/odata/v1/Products(<Id>)/$value

Authentication:
    Option A:
        CDSE_ACCESS_TOKEN

    Option B:
        CDSE_USERNAME
        CDSE_PASSWORD

Outputs:
    fallback_s1/raw/<city>/<selected_item_id>/
        <product>.zip
        product_query_result.json
        download_record.json
        optionally extracted .SAFE folder

Reports:
    qc/s1_fallback_downloads_copernicus/
        s1_fallback_copernicus_download_summary.csv/json/md

This script does not preprocess or merge S1. It only downloads the selected
fallback products.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


CATALOGUE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_ODATA_URL = "https://download.dataspace.copernicus.eu/odata/v1"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected fallback S1 products from Copernicus Data Space."
    )

    parser.add_argument(
        "--instance-root",
        type=str,
        required=True,
        help=(
            "Root of repaired instance, e.g. "
            "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired"
        ),
    )

    parser.add_argument(
        "--selection-csv",
        type=str,
        default=None,
        help=(
            "Selected fallback products CSV. If omitted, uses "
            "<instance-root>/qc/s1_fallback_selection/selected_s1_fallback_products.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Output root for downloads. If omitted, uses "
            "<instance-root>/fallback_s1/raw"
        ),
    )

    parser.add_argument(
        "--qc-dir",
        type=str,
        default=None,
        help=(
            "QC output dir. If omitted, uses "
            "<instance-root>/qc/s1_fallback_downloads_copernicus"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional cities to download. If omitted, all selected rows are used.",
    )

    parser.add_argument(
        "--catalogue-url",
        type=str,
        default=CATALOGUE_ODATA_URL,
        help=f"Copernicus catalogue OData URL. Default: {CATALOGUE_ODATA_URL}",
    )

    parser.add_argument(
        "--download-url",
        type=str,
        default=DOWNLOAD_ODATA_URL,
        help=f"Copernicus download OData URL. Default: {DOWNLOAD_ODATA_URL}",
    )

    parser.add_argument(
        "--auth-mode",
        choices=["token", "creds"],
        default="creds",
        help=(
            "token uses CDSE_ACCESS_TOKEN. "
            "creds uses CDSE_USERNAME and CDSE_PASSWORD to request a token."
        ),
    )

    parser.add_argument(
        "--token-env",
        type=str,
        default="CDSE_ACCESS_TOKEN",
        help="Environment variable containing Copernicus access token.",
    )

    parser.add_argument(
        "--username-env",
        type=str,
        default="CDSE_USERNAME",
        help="Environment variable containing Copernicus username/email.",
    )

    parser.add_argument(
        "--password-env",
        type=str,
        default="CDSE_PASSWORD",
        help="Environment variable containing Copernicus password.",
    )

    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only query product metadata; do not download.",
    )

    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Unzip downloaded ZIP products after download.",
    )

    parser.add_argument(
        "--delete-zip-after-unzip",
        action="store_true",
        help="Delete ZIP after successful extraction.",
    )

    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=16,
        help="Streaming download chunk size in MB. Default: 16.",
    )

    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Request timeout in seconds. Default: 120.",
    )

    parser.add_argument(
        "--max-download-retries",
        type=int,
        default=3,
        help="Number of download retries. Default: 3.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ZIP/metadata outputs.",
    )

    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def write_json(obj: Any, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["no_rows"])
        return

    fields: list[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "city",
        "selected_item_id",
        "product_name",
        "status",
        "product_id",
        "downloaded_zip_path",
        "safe_dir_count",
        "message",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Copernicus fallback S1 download summary\n\n")
        f.write(
            "This report summarizes Copernicus Data Space downloads for selected "
            "fallback Sentinel-1 GRD products.\n\n"
        )

        f.write("## Status counts\n\n")
        counts: dict[str, int] = {}
        for row in rows:
            status = row.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1

        for status, count in sorted(counts.items()):
            f.write(f"- `{status}`: {count}\n")

        f.write("\n## Downloads\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")

        for row in rows:
            values = []
            for col in cols:
                values.append(str(row.get(col, "")).replace("|", "/"))
            f.write("| " + " | ".join(values) + " |\n")


def load_selection_table(selection_csv: Path, cities: list[str] | None) -> pd.DataFrame:
    if not selection_csv.exists():
        raise FileNotFoundError(f"Selection CSV does not exist: {selection_csv}")

    df = pd.read_csv(selection_csv)

    required = {"city", "selected_item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Selection CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["city"] = df["city"].astype(str)
    df["selected_item_id"] = df["selected_item_id"].astype(str)

    if cities:
        df = df[df["city"].isin(cities)].copy()

    df = df[df["selected_item_id"].str.len() > 0].copy()

    if df.empty:
        raise ValueError("No selected fallback rows to process.")

    return df


def product_name_from_item_id(item_id: str) -> str:
    item_id = item_id.strip()

    if item_id.endswith(".SAFE"):
        return item_id

    if item_id.endswith(".zip"):
        item_id = item_id[:-4]

    return item_id + ".SAFE"


def get_access_token(args: argparse.Namespace) -> str:
    if args.auth_mode == "token":
        token = os.environ.get(args.token_env)
        if not token:
            raise RuntimeError(
                f"{args.token_env} is not set. Either set it or use --auth-mode creds."
            )
        return token

    username = os.environ.get(args.username_env)
    password = os.environ.get(args.password_env)

    if not username or not password:
        raise RuntimeError(
            f"{args.username_env} and/or {args.password_env} are not set."
        )

    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    response = requests.post(
        TOKEN_URL,
        data=data,
        timeout=args.request_timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to get Copernicus token. HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    token_json = response.json()

    token = token_json.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in token response: {token_json}")

    return token


def odata_string_literal(value: str) -> str:
    # OData single quote escaping: single quote becomes doubled.
    return "'" + value.replace("'", "''") + "'"


def query_product_by_name(
    product_name: str,
    catalogue_url: str,
    timeout: int,
) -> dict:
    # Exact query first.
    filter_expr = f"Name eq {odata_string_literal(product_name)}"
    encoded_filter = quote(filter_expr, safe="()' =")

    url = (
        f"{catalogue_url}/Products?"
        f"$filter={encoded_filter}"
        f"&$expand=Attributes,Locations"
    )

    response = requests.get(url, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(
            f"Copernicus OData query failed. HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()
    values = data.get("value", [])

    if values:
        return {
            "query_type": "exact_name",
            "query_url": url,
            "matches": values,
        }

    # Fallback contains query, useful if product is named with .zip or slightly different suffix.
    base = product_name.replace(".SAFE", "")
    filter_expr = f"contains(Name,{odata_string_literal(base)})"
    encoded_filter = quote(filter_expr, safe="()' ,=")

    url = (
        f"{catalogue_url}/Products?"
        f"$filter={encoded_filter}"
        f"&$expand=Attributes,Locations"
    )

    response = requests.get(url, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(
            f"Copernicus fallback OData query failed. HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()
    return {
        "query_type": "contains_name",
        "query_url": url,
        "matches": data.get("value", []),
    }


def select_best_product_match(product_name: str, matches: list[dict]) -> dict | None:
    if not matches:
        return None

    exact = [m for m in matches if str(m.get("Name", "")) == product_name]
    if exact:
        return exact[0]

    base = product_name.replace(".SAFE", "")
    starts = [m for m in matches if str(m.get("Name", "")).startswith(base)]
    if starts:
        return starts[0]

    return matches[0]


def content_disposition_filename(header_value: str | None) -> str | None:
    if not header_value:
        return None

    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', header_value)
    if match:
        return match.group(1)

    return None


def build_download_url(download_url: str, product_id: str) -> str:
    # CDSE docs use Products(<uuid>)/$value, without quotes.
    return f"{download_url}/Products({product_id})/$value"


def stream_download(
    url: str,
    output_zip: Path,
    token: str,
    chunk_size_bytes: int,
    timeout: int,
    max_retries: int,
    overwrite: bool,
) -> dict:
    if output_zip.exists() and not overwrite:
        return {
            "download_skipped_existing": True,
            "bytes_downloaded": output_zip.stat().st_size,
            "http_status": "",
            "content_type": "",
        }

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_zip.with_suffix(output_zip.suffix + ".part")

    if temp_path.exists():
        temp_path.unlink()

    headers = {
        "Authorization": f"Bearer {token}",
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=timeout,
            ) as response:
                status = response.status_code

                if status not in {200, 201, 202}:
                    raise RuntimeError(
                        f"Download failed HTTP {status}: {response.text[:1000]}"
                    )

                content_type = response.headers.get("Content-Type", "")
                bytes_written = 0

                with temp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bytes_written += len(chunk)

                if bytes_written == 0:
                    raise RuntimeError("Download wrote 0 bytes.")

                if output_zip.exists():
                    output_zip.unlink()

                temp_path.rename(output_zip)

                return {
                    "download_skipped_existing": False,
                    "bytes_downloaded": bytes_written,
                    "http_status": status,
                    "content_type": content_type,
                }

        except Exception as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()

            if attempt < max_retries:
                sleep_seconds = 5 * attempt
                print(
                    f"[WARN] Download attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {sleep_seconds}s...",
                    file=sys.stderr,
                )
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Download failed after {max_retries} attempts: {last_error}")


def unzip_product(
    zip_path: Path,
    extract_dir: Path,
    overwrite: bool,
    delete_zip_after_unzip: bool,
) -> dict:
    if not zip_path.exists():
        return {
            "safe_dirs": [],
            "unzip_errors": [f"ZIP does not exist: {zip_path}"],
        }

    safe_dirs_before = {str(p) for p in extract_dir.rglob("*.SAFE") if p.is_dir()}

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            safe_roots = sorted(
                {
                    m.split("/")[0]
                    for m in members
                    if ".SAFE/" in m or m.endswith(".SAFE")
                }
            )

            for root in safe_roots:
                existing = extract_dir / root
                if existing.exists() and overwrite:
                    shutil.rmtree(existing)

            zf.extractall(extract_dir)

        safe_dirs_after = {str(p) for p in extract_dir.rglob("*.SAFE") if p.is_dir()}
        new_safe_dirs = sorted(safe_dirs_after - safe_dirs_before)

        if not new_safe_dirs:
            # If already existed and overwrite=False, still report all SAFE dirs.
            new_safe_dirs = sorted(safe_dirs_after)

        if delete_zip_after_unzip and new_safe_dirs:
            zip_path.unlink()

        return {
            "safe_dirs": new_safe_dirs,
            "unzip_errors": [],
        }

    except Exception as exc:
        return {
            "safe_dirs": [],
            "unzip_errors": [str(exc)],
        }


def list_safe_dirs(download_dir: Path) -> list[str]:
    if not download_dir.exists():
        return []
    return [str(p) for p in sorted(download_dir.rglob("*.SAFE")) if p.is_dir()]


def process_row(
    row: dict,
    output_root: Path,
    token: str | None,
    args: argparse.Namespace,
) -> dict:
    city = str(row["city"])
    selected_item_id = str(row["selected_item_id"])
    product_name = product_name_from_item_id(selected_item_id)

    city_dir = output_root / city / selected_item_id
    city_dir.mkdir(parents=True, exist_ok=True)

    product_query_path = city_dir / "product_query_result.json"
    download_record_path = city_dir / "download_record.json"

    result = {
        "city": city,
        "selected_item_id": selected_item_id,
        "product_name": product_name,
        "download_dir": str(city_dir),
        "product_query_path": str(product_query_path),
        "download_record_path": str(download_record_path),
        "status": "",
        "message": "",
        "query_type": "",
        "product_id": "",
        "product_name_found": "",
        "product_online": "",
        "product_content_date_start": "",
        "product_content_date_end": "",
        "download_url": "",
        "downloaded_zip_path": "",
        "bytes_downloaded": "",
        "safe_dirs": "",
        "safe_dir_count": 0,
        "unzip_errors": "",
        "started_at_utc": now_utc(),
        "finished_at_utc": "",
    }

    try:
        query_result = query_product_by_name(
            product_name=product_name,
            catalogue_url=args.catalogue_url.rstrip("/"),
            timeout=args.request_timeout,
        )

        matches = query_result["matches"]
        product = select_best_product_match(product_name, matches)

        write_json(
            {
                "city": city,
                "selected_item_id": selected_item_id,
                "product_name": product_name,
                "query_result": query_result,
                "selected_product": product,
            },
            product_query_path,
            overwrite=True,
        )

        result["query_type"] = query_result["query_type"]

        if product is None:
            result["status"] = "error_product_not_found"
            result["message"] = "No Copernicus Data Space product found by name."
            write_json(result, download_record_path, overwrite=True)
            return result

        product_id = str(product.get("Id", ""))
        product_name_found = str(product.get("Name", ""))

        result["product_id"] = product_id
        result["product_name_found"] = product_name_found
        result["product_online"] = product.get("Online", "")
        result["product_content_date_start"] = (
            product.get("ContentDate", {}) or {}
        ).get("Start", "")
        result["product_content_date_end"] = (
            product.get("ContentDate", {}) or {}
        ).get("End", "")

        if not product_id:
            result["status"] = "error_product_id_missing"
            result["message"] = "Product found but Id is missing."
            write_json(result, download_record_path, overwrite=True)
            return result

        download_url = build_download_url(args.download_url.rstrip("/"), product_id)
        result["download_url"] = download_url

        zip_name = product_name_found
        if zip_name.endswith(".SAFE"):
            zip_name = zip_name + ".zip"
        elif not zip_name.endswith(".zip"):
            zip_name = zip_name + ".zip"

        output_zip = city_dir / zip_name
        result["downloaded_zip_path"] = str(output_zip)

        if args.metadata_only:
            safe_dirs = list_safe_dirs(city_dir)
            result["status"] = "metadata_only"
            result["message"] = "Product metadata found; download skipped."
            result["safe_dirs"] = ";".join(safe_dirs)
            result["safe_dir_count"] = len(safe_dirs)
            write_json(result, download_record_path, overwrite=True)
            return result

        if token is None:
            raise RuntimeError("No Copernicus access token available for download.")

        download_info = stream_download(
            url=download_url,
            output_zip=output_zip,
            token=token,
            chunk_size_bytes=args.chunk_size_mb * 1024 * 1024,
            timeout=args.request_timeout,
            max_retries=args.max_download_retries,
            overwrite=args.overwrite,
        )

        result["bytes_downloaded"] = download_info["bytes_downloaded"]

        safe_dirs = list_safe_dirs(city_dir)
        unzip_errors: list[str] = []

        if args.unzip:
            unzip_info = unzip_product(
                zip_path=output_zip,
                extract_dir=city_dir,
                overwrite=args.overwrite,
                delete_zip_after_unzip=args.delete_zip_after_unzip,
            )
            safe_dirs = unzip_info["safe_dirs"] or list_safe_dirs(city_dir)
            unzip_errors = unzip_info["unzip_errors"]

        result["safe_dirs"] = ";".join(safe_dirs)
        result["safe_dir_count"] = len(safe_dirs)
        result["unzip_errors"] = ";".join(unzip_errors)

        if unzip_errors:
            result["status"] = "downloaded_with_unzip_errors"
            result["message"] = "ZIP downloaded, but unzip failed."
        elif args.unzip and safe_dirs:
            result["status"] = "downloaded_safe_ready"
            result["message"] = "ZIP downloaded and SAFE directory extracted."
        elif output_zip.exists():
            result["status"] = "downloaded_zip_ready"
            result["message"] = "ZIP downloaded."
        else:
            result["status"] = "error_download_missing_file"
            result["message"] = "Download completed but ZIP file is missing."

        write_json(result, download_record_path, overwrite=True)
        return result

    except Exception as exc:
        result["status"] = "error"
        result["message"] = str(exc)
        write_json(result, download_record_path, overwrite=True)
        return result

    finally:
        result["finished_at_utc"] = now_utc()


def main() -> None:
    args = parse_args()

    instance_root = Path(args.instance_root)

    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root does not exist: {instance_root}")

    selection_csv = (
        Path(args.selection_csv)
        if args.selection_csv
        else instance_root / "qc" / "s1_fallback_selection" / "selected_s1_fallback_products.csv"
    )

    output_root = (
        Path(args.output_root)
        if args.output_root
        else instance_root / "fallback_s1" / "raw"
    )

    qc_dir = (
        Path(args.qc_dir)
        if args.qc_dir
        else instance_root / "qc" / "s1_fallback_downloads_copernicus"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    selection_df = load_selection_table(selection_csv, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] Selection CSV: {selection_csv}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] QC dir: {qc_dir}")
    print(f"[INFO] Rows to process: {len(selection_df)}")
    print(f"[INFO] auth_mode: {args.auth_mode}")
    print(f"[INFO] metadata_only: {args.metadata_only}")
    print(f"[INFO] unzip: {args.unzip}")

    token = None
    if not args.metadata_only:
        print("[INFO] Getting Copernicus access token...")
        token = get_access_token(args)

    rows: list[dict] = []

    for idx, row in enumerate(selection_df.to_dict(orient="records"), start=1):
        city = str(row["city"])
        selected_item_id = str(row["selected_item_id"])

        print(f"\n[STEP {idx}/{len(selection_df)}] {city}")
        print(f"[INFO] Selected item: {selected_item_id}")

        result = process_row(
            row=row,
            output_root=output_root,
            token=token,
            args=args,
        )

        rows.append(result)

        print(
            "[OK] "
            f"status={result['status']} | "
            f"product_id={result['product_id']} | "
            f"safe_dirs={result['safe_dir_count']} | "
            f"message={result['message']}"
        )

    csv_path = qc_dir / "s1_fallback_copernicus_download_summary.csv"
    json_path = qc_dir / "s1_fallback_copernicus_download_summary.json"
    md_path = qc_dir / "s1_fallback_copernicus_download_summary.md"

    write_csv(rows, csv_path, overwrite=args.overwrite)
    write_json(rows, json_path, overwrite=args.overwrite)
    write_markdown(rows, md_path, overwrite=args.overwrite)

    print("\n[DONE] Wrote:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    print("\n[SUMMARY]")
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()