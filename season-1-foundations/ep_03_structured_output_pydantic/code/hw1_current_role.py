# Add a current_role: str | None field to Resume and re-run. Did it extract correctly?

import os
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from pydantic import BaseModel,Field
from langchain.chat_models import init_chat_model


load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL","ollama:llama3.2")
# MODEL = "meta-llama/llama-prompt-guard-2-22m"
#MODEL = os.getenv("GROQ_MODEL","GROQ:meta-llama/llama-prompt-guard-2-22m")

SAMPLES_DIR = Path(__file__).parent / "samples"

class Resume(BaseModel):
    name :str = Field(description = "Full name of the candidate")
    email : str | None = Field(description = "Get email if present,else None")
    years_experience: float = Field(description = "Total year of professional experience")
    skills : list[str] = Field(description ="List of techinical skills mentions")
    current_role: str | None = Field(description = " Current Role of the Job.")

def read_sample(filename: str)-> str:
    """Read a fixed , shipped sample file a safely (no user-controlled paths)."""
    path = (SAMPLES_DIR / filename).resolve()
    if not str(path).startswith(str(SAMPLES_DIR.resolve())):
        raise ValueError("Invalid sample path")
    return path.read_text(encoding="utf-8")


def extract_resume(text: str)-> Resume | None :
    llm = init_chat_model(MODEL, temperature=0)
    structured_llm = llm.with_structured_output(Resume,include_raw = True)
    system = "You are a precise data extractor. Extract only what is present; never invent data "
    result = structured_llm.invoke([
        {"role":"system","content":system},
        {"role":"user","content":f"Extract structured data from this resume:\n\n{text}"},
    ])
    raw = result.get("raw")
    usage = getattr(raw,"usage_metadata",None)
    if usage:
        print(f"[dim]tokens - in : {usage.get('input_tokens')} "
            f"out : {usage.get("output_tokens")} total : {usage.get('total_tokens')}[/dim]")

    if result.get("parsing_error"):
        print("[red]Model output failed validtion; handle/retry instead of trusting it.[/red]")
        return None
    return result.get("parsed")

def main()-> None:
    text = read_sample("resume_messy.txt")
    print("[bold cyan] Extracting structured data from a messy resume ...[/bold cyan]\n")
    resume = extract_resume(text)

    if resume:
        print(resume.model_dump_json(indent=2))
        print(f"\n[green]Validated![/green] {len(resume.skills)} skills , "
               f"{resume.years_experience} yrs experience")


if __name__=="__main__":
    main()
