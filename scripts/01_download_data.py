"""
Script 1: Download SEC 10-K Filings

This script:
1. Fetches metadata from SEC for selected companies
2. Downloads the latest N years of 10-K filings
3. Saves raw HTML files
4. Cleans HTML to plain text
5. Creates a manifest CSV tracking all documents

SEC Rate Limits:
- 10 requests per second max
- We use 0.2 second delays to be conservative
"""

import os
import sys
import time
import re
import json
from typing import Dict, List, Optional
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    DATA_DIR,
    USER_AGENT,
    YEARS_PER_COMPANY,
    COMPANIES,
    SEC_SUBMISSIONS_URL,
    SEC_ARCHIVES_URL,
)

"""
    Convert HTML to clean plain text.
    
    Steps:
    1. Parse HTML with BeautifulSoup
    2. Remove script/style tags
    3. Extract text
    4. Normalize whitespace
    
    Args:
        html: Raw HTML string
        
    Returns:
        Cleaned plain text
"""
def clean_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" \n", "\n", text)
    
    return text.strip()

"""Fetch and parse JSON from URL."""
def fetch_json(url: str, headers: dict) -> dict:
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()

"""Fetch text content from URL."""
def fetch_text(url: str, headers: dict) -> str:    
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.text

"""
    Extract the latest N 10-K filings from SEC submissions JSON.
    
    The submissions JSON contains all filings for a company.
    We filter for form type "10-K" and sort by date.
    
    Args:
        submissions: SEC submissions JSON
        n: Number of filings to return
        
    Returns:
        List of filing metadata dicts
"""
def pick_latest_10k_filings(submissions: dict, n: int) -> List[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    filings = []
    for i in range(len(forms)):
        if forms[i] == "10-K":
            filings.append({
                "accessionNumber": accession_numbers[i],
                "filingDate": filing_dates[i],
                "primaryDocument": primary_docs[i],
                "reportDate": report_dates[i] if i < len(report_dates) else None,
            })

    filings.sort(key=lambda x: x["filingDate"], reverse=True)
    
    return filings[:n]

"""Remove leading zeros from CIK (SEC uses both formats)."""
def to_noleading_cik(cik_10: str) -> str:
    return str(int(cik_10))

"""Remove dashes from accession number for URL."""
def accession_no_dashes(acc: str) -> str:
    return acc.replace("-", "")

"""Main execution function."""
def main():
    print("SEC 10-K DOWNLOADER")
    print(f"User Agent: {USER_AGENT}")
    print(f"Companies: {len(COMPANIES)}")
    print(f"Years per company: {YEARS_PER_COMPANY}")
    print(f"Output: {RAW_DATA_DIR}")

    if "your.email@example.com" in USER_AGENT:
        print("\n WARNING: Please update USER_AGENT in src/config.py with your real email!")
        print("The SEC requires identification for automated access.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    }

    manifest_rows = []
    
    # Process each company
    for company, cik10 in tqdm(COMPANIES.items(), desc="Companies"):
        print(f"\n{'='*60}")
        print(f"Processing: {company} (CIK {cik10})")
        print(f"{'='*60}")
        
        try:
            sub_url = SEC_SUBMISSIONS_URL.format(cik=cik10)
            print(f"Fetching metadata from: {sub_url}")
            
            submissions = fetch_json(sub_url, headers)
            time.sleep(0.2)

            selected = pick_latest_10k_filings(submissions, YEARS_PER_COMPANY)
            
            if not selected:
                print(f"  No 10-K filings found for {company}")
                continue
            
            print(f"Found {len(selected)} 10-K filings")
            
            cik_nolead = to_noleading_cik(cik10)

            for filing in selected:
                acc = filing["accessionNumber"]
                filing_date = filing["filingDate"]
                primary_doc = filing["primaryDocument"]
                acc_nodash = accession_no_dashes(acc)

                doc_url = SEC_ARCHIVES_URL.format(
                    cik_nolead=cik_nolead,
                    accession_nodashes=acc_nodash,
                    primary_doc=primary_doc
                )
                
                print(f"  -> {filing_date}: {primary_doc}")

                base_name = f"{company}_{cik10}_{filing_date}_10K_{acc_nodash}"
                raw_fname = f"{base_name}.html"
                clean_fname = f"{base_name}.txt"
                
                raw_path = RAW_DATA_DIR / raw_fname
                clean_path = CLEANED_DATA_DIR / clean_fname

                raw_html = fetch_text(doc_url, {"User-Agent": USER_AGENT})
                
                with open(raw_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(raw_html)

                cleaned_text = clean_text_from_html(raw_html)
                
                with open(clean_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(cleaned_text)

                manifest_rows.append({
                    "company": company,
                    "cik": cik10,
                    "form": "10-K",
                    "filing_date": filing_date,
                    "accession_number": acc,
                    "primary_document": primary_doc,
                    "source_url": doc_url,
                    "local_raw_path": str(raw_path),
                    "local_clean_path": str(clean_path),
                    "file_size_kb": round(len(cleaned_text) / 1024, 2),
                })
                
                time.sleep(0.2)
        
        except Exception as e:
            print(f"Error processing {company}: {e}")
            continue

    manifest_path = DATA_DIR / "manifest.csv"
    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_path, index=False)

    print(" DOWNLOAD COMPLETE")
    print(f"Total filings downloaded: {len(df)}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Raw files: {RAW_DATA_DIR}")
    print(f"Cleaned files: {CLEANED_DATA_DIR}")
    print("\nSample manifest:")
    print(df.head())


if __name__ == "__main__":
    main()
