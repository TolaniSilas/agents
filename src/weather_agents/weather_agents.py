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

Use the get_lat_long tool to obtain the latitude and longitude of the user's location.
Then use the get_weather tool to retrieve accurate weather information for those coordinates.

You must:

Respond in simple, clear language that anyone can understand.
Return key weather details, including: temperature and weather condition.
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
                      instrument=True
                      )


@weather_agent.tool
async def get_lat_long(
    ctx: RunContext[Dependencies], location_description: str
    ) -> dict[str, float]:
    """Get the latitude and longitude of a location.
    
    Args:
        ctx: The context object containing the dependencies.
        location_description: Description of the location to get the latitude and longitude for.
    """

    if not ctx.deps.geo_api_key:
        # if the API key is not set, return a dummy response(London)
        return {'lat': 51.1, 'lng': -0.1}
    
    params = {
        "query": location_description,
        "api_key": ctx.deps.geo_api_key
    }
    
    with logfire.span("Calling geocode API", params=params) as span:
        response = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
        response.raise_for_status()
        data = response.json()
        span.set_attribute('response', data)
        
        if data:
            return {"lat": data[0]["lat"], "long": data[0]["lon"]}
        
        else:
            raise ModelRetry('Could not find the location!')
        

@weather_agent.tool
async def get_weather(
    ctx: RunContext[Dependencies], lat: float, long: float
) -> dict[str, Any]:
    """ Get the weather for a given latitude and longitude.
    
    Args: 
    ctx: The context object containing the dependencies.
    lat: The latitude of the location.
    long: The longitude of the location.
    """

    if not ctx.deps.weather_api_key:
        # if the API key is not set, return a dummy response.
        return {
            "temperature": 20,
            "weather_condition": "Sunny"
        }
    
    params = {
        "apikey": ctx.deps.weather_api_key,
        "location": f"{lat}, {long}",
        "units": "metric"
    }

    with logfire.span("Calling weather API", params=params) as span:
        response = await ctx.deps.client.get(
            "https://api.tomorrow.io/v4/weather/realtime", params=params)
        
        response.raise_for_status()
        data = response.json()
        span.set_attribute("response", data)

    values = data["data"]["values"]

    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }

    return {
        "temperature": f'{values["temperatureApparent"]:0.0f}°C',
        "weather_condition": code_lookup.get(values["weatherCode"], "Unknown")
    }


async def main():
    async with AsyncClient() as client:
        # Retrieve the API keys from environment variables.
        weather_api_key = os.getenv("WEATHER_API_KEY")
        geo_api_key = os.getenv("GEO_API_KEY")

        deps = Dependencies(
            client=client,
            weather_api_key=weather_api_key,
            geo_api_key=geo_api_key
        )

        result = await weather_agent.run(
            user_prompt='What is the weather like in Lagos?', deps=deps
            )
        
        debug(result)
        print('Response:', result.output)


if __name__ == "__main__":
    asyncio.run(main())



