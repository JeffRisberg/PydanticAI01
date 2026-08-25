import os
import random

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

load_dotenv()

agent = Agent(
  'openai:gpt-5',
  deps_type=str,
  instructions=(
      "You're a dice game, you should roll the die and see if the number "
      "you get back matches the user's guess. If so, tell them they're a winner. "
      "Use the player's name in the response."
  ),
)


@agent.tool_plain  
def roll_dice() -> str:
  """Roll a six-sided die and return the result."""
  return str(random.randint(1, 6))


@agent.tool  
def get_player_name(ctx: RunContext[str]) -> str:
  """Get the player's name."""
  return ctx.deps


dice_result = agent.run_sync('My guess is 4', deps='Jeff')  
print(dice_result.output)
#> Congratulations Jeff, you guessed correctly! You're a winner!

