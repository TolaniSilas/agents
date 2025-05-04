import os 
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any
import logfire
from httpx import AsyncClient
from devtools import debug
from pydantic_ai import Agent, RunContext, ModelRetry


# Load the environment variables from the .env file.
load_dotenv()


logfire.configure(send_to_logfire="if-token-present")

@dataclass
class Dependencies:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


# Define the system prompt for the agent.
system_prompt = """
You are a smart and helpful Weather Assistant.
Your job is to:

Be concise — reply with one sentence.

Use the get_lat_lng tool to obtain the latitude and longitude of the user’s location.
Then use the get_weather tool to retrieve accurate weather information for those coordinates.

You must:

Respond in simple, clear language that anyone can understand.
Return key weather details, including: temperature, weather condition, humidity, and wind speed.
If any required data is missing or unavailable, politely inform the user.

Your responses should always:

Begin with a brief summary, e.g., “The weather in Lagos is sunny and 30°C.”
Optionally include a structured JSON output for programmatic use (if requested).
Avoid technical jargon unless the user specifically asks for it.
"""

weather_agent = Agent(model="groq:llama-3.3-70b-versatile", 
                      system_prompt=system_prompt,
                      deps_type=Dependencies,
                      retries=2,
                      instrument=2
                      )



