from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()

# 1. Define the Structured Output Schema
class DimensionScore(BaseModel):
    score: int = Field(..., description="Score from 1 to 5, where 5 is perfect.")
    reasoning: str = Field(..., description="Detailed explanation for why this score was given.")
    missing_elements: List[str] = Field(..., description="Specific items or context missing from the explanation.")

class DocumentAssessment(BaseModel):
    coverage: DimensionScore = Field(..., description="Evaluation of how completely the explanation covers the specification.")
    fidelity: DimensionScore = Field(..., description="Evaluation of the accuracy and lack of contradictions.")
    abstraction_quality: DimensionScore = Field(..., description="Evaluation of how effectively it simplifies the concept for the target audience.")
    overall_verdict: str = Field(..., description="Final conclusion on whether the explanation document is reliable and fit for use.")

# 2. Dependencies passed to the agent for this run
@dataclass
class DocumentPair:
    spec_text: str
    explanation_text: str

# 3. Define the Agent
agent = Agent(
    "openai:gpt-4o",
    deps_type=DocumentPair,
    output_type=DocumentAssessment,
    instructions=(
        "You are an expert technical auditor. Your task is to evaluate whether an Explanation Document "
        "is an accurate, comprehensive, and helpful breakdown of a behavior outlined in a Specification Document. "
        "Use the provided tools to fetch both documents and to check specific terms before scoring. "
        "Review both texts carefully and score them strictly based on the requested output schema."
    ),
)

@agent.tool
def get_specification_document(ctx: RunContext[DocumentPair]) -> str:
    """Fetch the full text of the specification document being audited against."""
    return ctx.deps.spec_text

@agent.tool
def get_explanation_document(ctx: RunContext[DocumentPair]) -> str:
    """Fetch the full text of the explanation document under review."""
    return ctx.deps.explanation_text

@agent.tool
def count_term_occurrences(ctx: RunContext[DocumentPair], document: str, term: str) -> int:
    """Count case-insensitive occurrences of a term in a document.

    Args:
        document: Which document to search, either "specification" or "explanation".
        term: The word or phrase to count.
    """
    text = ctx.deps.spec_text if document == "specification" else ctx.deps.explanation_text
    return text.lower().count(term.lower())

def assess_explanation(spec_text: str, explanation_text: str) -> DocumentAssessment:
    deps = DocumentPair(spec_text=spec_text, explanation_text=explanation_text)
    result = agent.run_sync(
        "Audit the explanation document against the specification document and produce the assessment.",
        deps=deps,
    )
    return result.output

# --- Example Usage Run ---
if __name__ == "__main__":
    mock_spec = """
    Authentication System Spec: The system must allow users to log in using an email and password.
    If login fails 3 times sequentially, the account must lock for exactly 15 minutes.
    An email notification must immediately be dispatched to the user upon account lockout.
    """

    mock_explanation = """
    How Login Works: Users type in their email address and secret password to gain access to the application.
    If a user types the wrong password too many times, our security guard system temporarily stops them
    from logging back in to keep hackers out.
    """

    print("Analyzing documents...")
    assessment_result = assess_explanation(mock_spec, mock_explanation)

    print("\n--- Assessment Results ---")
    print(f"Coverage Score: {assessment_result.coverage.score}/5")
    print(f"Coverage Reasoning: {assessment_result.coverage.reasoning}")
    print(f"Coverage Missing: {assessment_result.coverage.missing_elements}\n")

    print(f"Fidelity Score: {assessment_result.fidelity.score}/5")
    print(f"Fidelity Reasoning: {assessment_result.fidelity.reasoning}\n")

    print(f"Overall Verdict: {assessment_result.overall_verdict}")
