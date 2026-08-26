"""Plan-and-execute agent example: a trip-planning assistant.

Three agents cooperate:
  1. `planner_agent` decomposes a high-level trip request into an ordered
     list of concrete research steps (structured output: `TripPlan`).
  2. `executor_agent` carries out each step in turn using tools that look up
     flights, hotels, weather, and local costs (mock data).
  3. `synthesizer_agent` combines all step findings into a structured
     `TripItinerary`, including a budget check.
"""

from dataclasses import dataclass
from datetime import date
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()


# ---------------------------------------------------------------------------
# Domain data
# ---------------------------------------------------------------------------

@dataclass
class TripRequest:
    origin: str
    destination: str
    start_date: date
    end_date: date
    budget_usd: float


# Mock data stores, keyed by destination, standing in for real travel APIs.
_FLIGHT_PRICES = {"Tokyo": 950, "Lisbon": 620, "Nairobi": 880, "London": 990}
_HOTEL_NIGHTLY_RATES = {"Tokyo": 180, "Lisbon": 110, "Nairobi": 90, "London": 200}
_DAILY_LOCAL_COST = {"Tokyo": 70, "Lisbon": 45, "Nairobi": 35, "London": 47}
_WEATHER_FORECAST = {
    "Tokyo": "Mild, 18-23C, occasional rain",
    "Lisbon": "Sunny, 22-27C, dry",
    "Nairobi": "Mild, 16-24C, short afternoon showers",
    "London": "Cloudy, 22-27C, rain often",
}
_VACCINATION_INFO = {
    "Tokyo": "no",
    "Lisbon": "no",
    "Nairobi": "yes",
    "London": "no",
}


# ---------------------------------------------------------------------------
# Planner agent: goal -> ordered plan
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    step_number: int = Field(..., description="1-based order in which the step should be executed.")
    description: str = Field(...,
                             description="Concrete, actionable instruction for the executor, e.g. 'Look up hotel rates in Lisbon'.")
    rationale: str = Field(..., description="Why this step is needed to fulfill the trip request.")


class TripPlan(BaseModel):
    goal: str = Field(..., description="Restatement of the user's trip goal.")
    steps: List[PlanStep] = Field(..., description="Ordered steps needed to research and validate the trip.")


planner_agent = Agent(
    "openai:gpt-4o",
    deps_type=TripRequest,
    output_type=TripPlan,
    instructions=(
        "You are a trip-planning strategist. Given a trip request, break the work into an ordered "
        "list of concrete research steps an assistant should carry out before booking, such as checking "
        "flight prices, hotel rates, weather, vaccination, and daily living costs. "
        "Each step should be independently actionable and reference the destination explicitly. "
        "Do not book anything or invent prices yourself."
    ),
)


@planner_agent.tool
def get_trip_request(ctx: RunContext[TripRequest]) -> str:
    """Return the raw trip request details to ground the plan."""
    req = ctx.deps
    return (
        f"Origin: {req.origin}, Destination: {req.destination}, "
        f"Dates: {req.start_date} to {req.end_date}, Budget: ${req.budget_usd:.0f}"
    )


# ---------------------------------------------------------------------------
# Executor agent: carries out a single plan step using domain tools
# ---------------------------------------------------------------------------

executor_agent = Agent(
    "openai:gpt-4o",
    deps_type=TripRequest,
    output_type=str,
    instructions=(
        "You are a travel research assistant executing one step of a larger trip plan. "
        "Use the available tools to gather real figures rather than guessing. "
        "Respond with a short factual summary of what you found for this step."
    ),
)


@executor_agent.tool
def search_flights(ctx: RunContext[TripRequest]) -> str:
    """Look up the round-trip flight price for the trip's origin and destination."""
    price = _FLIGHT_PRICES.get(ctx.deps.destination, 750)
    return f"Round-trip {ctx.deps.origin}->{ctx.deps.destination}: ${price}"


@executor_agent.tool
def search_hotels(ctx: RunContext[TripRequest]) -> str:
    """Look up nightly hotel rates and compute total lodging cost for the stay."""
    nights = (ctx.deps.end_date - ctx.deps.start_date).days
    rate = _HOTEL_NIGHTLY_RATES.get(ctx.deps.destination, 120)
    return f"{nights} nights at ${rate}/night = ${nights * rate} in {ctx.deps.destination}"


@executor_agent.tool
def get_weather_forecast(ctx: RunContext[TripRequest]) -> str:
    """Look up the general weather forecast for the destination."""
    return _WEATHER_FORECAST.get(ctx.deps.destination, "Forecast unavailable")


@executor_agent.tool
def estimate_local_costs(ctx: RunContext[TripRequest]) -> str:
    """Estimate total food/transport/activity costs for the length of the stay."""
    days = (ctx.deps.end_date - ctx.deps.start_date).days
    daily = _DAILY_LOCAL_COST.get(ctx.deps.destination, 50)
    return f"{days} days at ${daily}/day = ${days * daily} in local expenses"


@executor_agent.tool
def check_vaccination(ctx: RunContext[TripRequest]) -> str:
    """Look up the vaccination requirements for the destination."""
    requirements = _VACCINATION_INFO.get(ctx.deps.destination, "no")
    return f"Vaccination: ${requirements}"


# ---------------------------------------------------------------------------
# Combiner agent: combine step results into a final structured itinerary
# ---------------------------------------------------------------------------

class TripItinerary(BaseModel):
    summary: str = Field(..., description="Short human-readable overview of the plan.")
    estimated_total_cost: float = Field(..., description="Sum of flight, hotel, and local costs in USD.")
    within_budget: bool = Field(..., description="Whether estimated_total_cost is within the requester's budget.")
    vaccination_requirements: bool = Field(..., description="Whether vaccination is required.")
    highlights: List[str] = Field(..., description="Notable findings, e.g. weather or standout costs.")


combiner_agent = Agent(
    "openai:gpt-4o",
    deps_type=TripRequest,
    output_type=TripItinerary,
    instructions=(
        "You are finalizing a trip plan. You will be given the research findings gathered for each "
        "planning step. Sum the relevant costs, compare against the requester's budget, and produce "
        "a structured itinerary summary."
    ),
)


# ---------------------------------------------------------------------------
# Orchestration: plan -> execute each step -> combine
# ---------------------------------------------------------------------------

def plan_and_execute_trip(request: TripRequest) -> TripItinerary:
    plan_result = planner_agent.run_sync(
        f"Plan the research needed for a trip to {request.destination}.", deps=request
    )
    plan = plan_result.output
    print(f"\nPlan: {plan}")
    for step in plan.steps:
        print(f"{step}")

    step_findings: List[str] = []
    for step in plan.steps:
        step_result = executor_agent.run_sync(step.description, deps=request)
        step_findings.append(f"Step {step.step_number} ({step.description}): {step_result.output}")
    print(f"\nStep findings:")
    for finding in step_findings:
          print(f"{finding}")

    findings_text = "\n".join(step_findings)
    itinerary_result = combiner_agent.run_sync(
        f"Trip goal: {plan.goal}\nBudget: ${request.budget_usd:.0f}\n\nResearch findings:\n{findings_text}",
        deps=request,
    )
    return itinerary_result.output


# --- Example Usage Run ---
if __name__ == "__main__":
    trip_request = TripRequest(
        origin="Seattle",
        destination="Nairobi",
        start_date=date(2026, 10, 5),
        end_date=date(2026, 10, 12),
        budget_usd=2000,
    )

    print(f"Planning trip to {trip_request.destination}...")
    itinerary = plan_and_execute_trip(trip_request)

    print("\n--- Final Itinerary ---")
    print(f"Summary: {itinerary.summary}")
    print(f"Estimated total cost: ${itinerary.estimated_total_cost:.2f}")
    print(f"Within budget: {itinerary.within_budget}")
    print(f"Vaccination required: {itinerary.vaccination_requirements}")
    print("Highlights:")
    for h in itinerary.highlights:
        print(f"  - {h}")
