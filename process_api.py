
import os
import json
import time
import logging
import re
from io import BytesIO
from typing import Any
from fastapi.responses import Response
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader
from groq import Groq

from employee_data_agent import extract_employee_data


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process_api")

app = FastAPI(
    title="Employee and ESG Reporting API",
    description="Employee data extraction and ESG report summarization",
    version="3.0.0",
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

EMPLOYEE_DATA_TABLE = os.getenv(
    "EMPLOYEE_DATA_TABLE", "employee_data_files"
)
ESG_FILES_TABLE = os.getenv("ESG_FILES_TABLE", "esg_files")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "bput")

# One complete report is sent in one Groq request.
# Increase only if the model's available token budget permits.
ESG_REPORT_TEXT_LIMIT = int(
    os.getenv("ESG_REPORT_TEXT_LIMIT", "12000")
)

GROQ_MAX_ATTEMPTS = 4

SUPABASE_HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}

groq_client = (
    Groq(
        api_key=GROQ_API_KEY,
        max_retries=0,
        timeout=120.0,
    )
    if GROQ_API_KEY
    else None
)


# =========================================================
# REQUEST MODELS
# =========================================================

class EmployeeExtractRequest(BaseModel):
    company_name: str = Field(..., min_length=1)


class ESGReportRequest(BaseModel):
    report_name: str = Field(..., min_length=1)


# =========================================================
# CONFIGURATION VALIDATION
# =========================================================

def check_configuration():
    missing = []

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")

    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if missing:
        raise HTTPException(
            status_code=500,
            detail="Missing configuration: " + ", ".join(missing),
        )


# =========================================================
# SUPABASE HELPERS
# =========================================================

def supabase_request(method: str, path: str, **kwargs):
    check_configuration()

    url = f"{SUPABASE_URL}/rest/v1/{path}"

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=SUPABASE_HEADERS,
            timeout=60,
            **kwargs,
        )
    except requests.RequestException as exc:
        logger.exception("Supabase request failed")

        raise HTTPException(
            status_code=502,
            detail="Could not connect to Supabase.",
        ) from exc

    if not response.ok:
        logger.error(
            "Supabase HTTP %s: %s",
            response.status_code,
            response.text[:1500],
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Supabase returned HTTP {response.status_code}. "
                "Check the server logs."
            ),
        )

    if not response.content:
        return None

    try:
        return response.json()
    except ValueError:
        return response.text


def fetch_latest_esg_record(report_name: str) -> dict[str, Any]:
    """Fetch only the newest row matching the report name."""

    rows = supabase_request(
        "GET",
        ESG_FILES_TABLE,
        params={
            "select": "id,report_name,url,created_at",
            "report_name": f"eq.{report_name}",
            "order": "created_at.desc",
            "limit": "1",
        },
    )

    if not isinstance(rows, list) or not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No ESG report found named '{report_name}'.",
        )

    record = rows[0]

    if record.get("id") is None:
        raise HTTPException(
            status_code=500,
            detail="The selected ESG record has no id.",
        )

    if not record.get("url"):
        raise HTTPException(
            status_code=422,
            detail="The selected ESG record has no PDF URL.",
        )

    logger.info(
        "Selected ESG record id=%s created_at=%s",
        record["id"],
        record.get("created_at"),
    )

    return record


def save_report_json(
    record_id: Any,
    report_json: dict[str, Any],
):
    """Update only the selected database row."""

    supabase_request(
        "PATCH",
        ESG_FILES_TABLE,
        params={
            "id": f"eq.{record_id}",
        },
        json={
            "report_json": report_json,
        },
    )

    logger.info("Saved report JSON for record id=%s", record_id)


# =========================================================
# PDF HELPERS
# =========================================================

def download_pdf(file_url: str) -> bytes:
    """Download a PDF using its HTTP URL or Supabase object path."""

    if not file_url or not file_url.strip():
        raise HTTPException(
            status_code=422,
            detail="The PDF URL is empty.",
        )

    url = file_url.strip()

    try:
        if url.startswith(("https://", "http://")):
            response = requests.get(
                url,
                timeout=90,
                headers={
                    "User-Agent": "ESG-Reporting-Service/1.0"
                },
            )
        else:
            object_path = url.lstrip("/")
            bucket_prefix = f"{SUPABASE_STORAGE_BUCKET}/"

            if object_path.startswith(bucket_prefix):
                object_path = object_path[len(bucket_prefix):]

            response = requests.get(
                f"{SUPABASE_URL}/storage/v1/object/"
                f"{SUPABASE_STORAGE_BUCKET}/{object_path}",
                headers=SUPABASE_HEADERS,
                timeout=90,
            )

        response.raise_for_status()
        pdf_bytes = response.content

    except requests.RequestException as exc:
        logger.exception("PDF download failed")

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not download the PDF. Check its URL, "
                "signed URL expiry, bucket and permissions."
            ),
        ) from exc

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=422,
            detail="The downloaded file is not a recognizable PDF.",
        )

    return pdf_bytes


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from all readable PDF pages."""

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        page_texts = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                page_texts.append(
                    f"\n--- PAGE {page_number} ---\n{text}"
                )

        full_text = "\n".join(page_texts).strip()

    except Exception as exc:
        logger.exception("PDF text extraction failed")

        raise HTTPException(
            status_code=422,
            detail=f"Could not read the PDF: {exc}",
        ) from exc

    if not full_text:
        raise HTTPException(
            status_code=422,
            detail=(
                "No readable text was found in the PDF. "
                "Scanned PDFs require OCR."
            ),
        )

    logger.info(
        "Extracted %s characters from PDF",
        len(full_text),
    )

    return full_text


# =========================================================
# GROQ RETRY HANDLING
# =========================================================

def call_groq_with_retry(messages: list[dict[str, str]]) -> str:
    """Call Groq with bounded retries for rate limits."""

    if groq_client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured.",
        )

    for attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Groq returned an empty response.")

            return content

        except Exception as exc:
            status_code = getattr(exc, "status_code", None)

            # Only retry rate limits and temporary server failures.
            retryable = (
                status_code == 429
                or (
                    isinstance(status_code, int)
                    and status_code >= 500
                )
            )

            if not retryable or attempt == GROQ_MAX_ATTEMPTS:
                logger.exception(
                    "Groq request failed on attempt %s",
                    attempt,
                )

                if status_code == 429:
                    raise HTTPException(
                        status_code=429,
                        detail=(
                            "Groq rate limit persists after retries. "
                            "Wait before submitting the report again."
                        ),
                    ) from exc

                raise HTTPException(
                    status_code=502,
                    detail=f"Groq request failed: {exc}",
                ) from exc

            wait_seconds = None
            body = getattr(exc, "body", None)

            if isinstance(body, dict):
                error_data = body.get("error", {})
                message = error_data.get("message", "")

                match = re.search(
                    r"try again in ([0-9.]+)s",
                    message,
                    re.IGNORECASE,
                )

                if match:
                    wait_seconds = float(match.group(1))

            if wait_seconds is None:
                wait_seconds = min(5 * attempt, 30)

            # Add a small buffer to the suggested wait.
            wait_seconds = min(wait_seconds + 1, 60)

            logger.warning(
                "Groq rate limit/temporary error. "
                "Waiting %.1f seconds before retry %s/%s.",
                wait_seconds,
                attempt,
                GROQ_MAX_ATTEMPTS,
            )

            time.sleep(wait_seconds)

    raise HTTPException(
        status_code=502,
        detail="Groq request failed after retries.",
    )


# =========================================================
# ESG REPORT SUMMARIZATION
# =========================================================

def parse_json_response(response_text: str) -> dict[str, Any]:
    """Parse Groq output and report malformed JSON clearly."""

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        start = max(0, exc.pos - 200)
        end = min(len(response_text), exc.pos + 200)

        logger.error(
            "Invalid JSON from Groq: line=%s column=%s "
            "error=%s nearby=%r",
            exc.lineno,
            exc.colno,
            exc.msg,
            response_text[start:end],
        )

        raise ValueError(
            f"Invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("Groq must return a JSON object.")

    return result


ESG_SUMMARY_SYSTEM_PROMPT = """
You are an ESG and BRSR report summarization assistant.

Your task is to summarize the complete report into ONE concise JSON object.

IMPORTANT RULES:
1. Return valid JSON only, with no Markdown fences.
2. Use only facts supported by the supplied report.
3. Never invent company details, metrics, achievements or targets.
4. Preserve important numerical values, units and reporting periods.
5. Clearly distinguish actual performance from future targets.
6. Summarize the nine BRSR principles where evidence is available.
7. If the report does not cover a principle, state that evidence was
   not found in the supplied report text.
8. Do not create a separate summary for every page or section.
9. Avoid duplicate information and unnecessary detail.
10. Keep the final report summary concise and useful for management.

Use this JSON structure:

{
  "report_name": "",
  "executive_summary": "",
  "organization_profile": {
    "organization_name": null,
    "reporting_period": null,
    "industry": null,
    "reporting_scope": null
  },
  "key_esg_highlights": [],
  "environmental_performance": {
    "key_metrics": [],
    "initiatives": [],
    "targets": []
  },
  "social_performance": {
    "key_metrics": [],
    "initiatives": [],
    "targets": []
  },
  "governance_performance": {
    "key_metrics": [],
    "policies_and_controls": [],
    "material_issues": []
  },
  "brsr_principles": [
    {
      "principle": 1,
      "name": "Ethics, Transparency and Accountability",
      "summary": "",
      "key_metrics": [],
      "targets": []
    },
    {
      "principle": 2,
      "name": "Sustainable and Safe Goods and Services",
      "summary": "",
      "key_metrics": [],
      "targets": []
    },
    {
      "principle": 3,
      "name": "Employee Well-being",
      "summary": "",
      "key_metrics": [],
      "targets": []
    },
    {
      "principle": 4,
      "name": "Stakeholder Engagement",
      "summary": "",
      "key_metrics": [],
      "targets": []
    },
    {
      "principle": 5,
      "name": "Human Rights",
      "summary": "",
      "key_metrics": [],
      "targets": []
    },
    {
      "principle": 6,
      "name": "Environment",
      "summary": "",
      "key_metrics": [],
      "targets": []
    },
    {
      "principle": 7,
      "name": "Responsible Public Policy Advocacy",
      "summary": "",
      "key_metrics": [],
      "targets": []
    },
    {
      "principle": 8,
      "name": "Inclusive Growth and Equitable Development",
      "summary": "",
      "key_metrics": [],
      "targets": []
    },
    {
      "principle": 9,
      "name": "Consumer Value",
      "summary": "",
      "key_metrics": [],
      "targets": []
    }
  ],
  "sdg_alignment": [],
  "risks_and_opportunities": [],
  "reporting_gaps": [],
  "conclusion": ""
}

For key metrics, use concise objects with metric name, value, unit,
period and status where available.
Keep lists short and relevant.
Do not claim regulatory compliance unless the report provides evidence.
"""


def extract_report_json(
    report_name: str,
    pdf_text: str,
) -> dict[str, Any]:
    """
    Send the report text in ONE request and return ONE summary JSON.
    This does not split the report into chunks.
    """

    if not pdf_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No report text is available for summarization.",
        )

    if len(pdf_text) > ESG_REPORT_TEXT_LIMIT:
        logger.warning(
            "Report text is %s characters; only the first %s "
            "characters will be summarized.",
            len(pdf_text),
            ESG_REPORT_TEXT_LIMIT,
        )

    report_text = pdf_text[:ESG_REPORT_TEXT_LIMIT]

    messages = [
        {
            "role": "system",
            "content": ESG_SUMMARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Report name: {report_name}\n\n"
                "Summarize the following report as one concise JSON object. "
                "Consider the report as a whole, retaining the most "
                "important findings and metrics.\n\n"
                "--- BEGIN COMPLETE REPORT TEXT ---\n"
                f"{report_text}\n"
                "--- END COMPLETE REPORT TEXT ---"
            ),
        },
    ]

    logger.info(
        "Generating one ESG summary for '%s' using %s characters",
        report_name,
        len(report_text),
    )

    # One summary request, not a sequence of chunk extraction calls.
    content = call_groq_with_retry(messages)

    try:
        result = parse_json_response(content)
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Groq returned invalid summary JSON: {exc}",
        ) from exc

    result["report_name"] = report_name

    return result


# =========================================================
# API ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Employee and ESG Reporting API is running",
        "routes": [
            "/health",
            "/extract",
            "/extract-report",
            "/docs",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase_configured": bool(
            SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
        ),
        "groq_configured": bool(GROQ_API_KEY),
    }


@app.post("/extract")
def extract_employee(request: EmployeeExtractRequest):
    """Existing employee extraction endpoint."""

    check_configuration()

    try:
        return extract_employee_data(request.company_name)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Employee extraction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Employee extraction failed: {exc}",
        ) from exc


@app.post("/extract-report")
def extract_esg_report(request: ESGReportRequest):
    """
    Fetch the newest matching ESG report, summarize it in one Groq
    request, save the summary JSON, and return it.
    """

    check_configuration()

    # 1. Select the newest record with this report name.
    record = fetch_latest_esg_record(request.report_name)

    # 2. Download the PDF.
    pdf_bytes = download_pdf(record["url"])

    # 3. Extract all selectable PDF text.
    pdf_text = extract_pdf_text(pdf_bytes)

    # 4. Generate one concise JSON summary.
    report_json = extract_report_json(
        report_name=record.get(
            "report_name",
            request.report_name,
        ),
        pdf_text=pdf_text,
    )

    # 5. Update only the selected record.
    save_report_json(
        record_id=record["id"],
        report_json=report_json,
    )

    # 6. Return the summary.
    return {
        "success": True,
        "report_id": record["id"],
        "report_name": record.get(
            "report_name",
            request.report_name,
        ),
        "created_at": record.get("created_at"),
        "json_saved": True,
        "report_json": report_json,
    }


@app.get("/fetch-report-pdf")
def fetch_report_pdf(report_name: str):
    """
    Return the PDF URL stored in the latest matching ESG record.
    """
    check_configuration()

    record = fetch_latest_esg_record(report_name)

    return {
        "success": True,
        "report_id": record["id"],
        "report_name": record.get("report_name"),
        "pdf_url": record["url"],
    }
