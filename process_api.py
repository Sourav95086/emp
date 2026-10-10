import os
import secrets
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from employee_data_agent import EmployeeAgentError, extract_employee_data

app = FastAPI(
    title="Employee Data Extraction API",
    description="Fetches a company's employee PDF from Supabase and returns extracted JSON.",
    version="1.0.0",
)


class ExtractRequest(BaseModel):
    company_name: str = Field(..., min_length=1, description="Exact company name in employee_data_files")


def check_api_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv("PROCESS_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="Server configuration error: PROCESS_API_KEY is not configured.")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/")
def root():
    return {
        "service": "Employee Data Extraction API",
        "status": "running",
        "endpoints": {"health": "GET /health", "extract": "POST /extract"},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract(request: ExtractRequest):
    try:
        return extract_employee_data(request.company_name.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmployeeAgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unexpected employee extraction error."
        )
