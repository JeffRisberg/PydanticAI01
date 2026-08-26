# PydanticAI01

A small example project using [pydantic-ai](https://ai.pydantic.dev/) to build an agent (a dice-guessing game) backed by an OpenAI model.

## Setup

1. Create and activate a virtual environment, then update pip:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -U pip
   ```

2. Install dependencies with pip:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:

   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

## Running

```bash
python main.py
```

This runs the dice-game agent, which rolls a die, checks it against a guess, and prints the result.
