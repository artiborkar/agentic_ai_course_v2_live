from dotenv import load_dotenv
from typing import Literal
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def get_weather(city:str , unit: Literal["celsius","fahrenheit"] = "celsius" ) -> str:
    """ The function will return the weather of the city you provide"""
    data = {
            "Delhi" : "35°C, Sunny",
            "Mumbai " :"30°C , Honid",
            "Bangalore" : "26°C , Cloudy",
            "Pune" : "28°C , Clear"

    }

    return f"Weather in  {city} : {data.get(city,'Not Available')} ({unit})"

llm  = ChatOllama(model="llama3.2",temperature=0).bind_tools([get_weather])

ai_msg = llm.invoke([HumanMessage(content="What is the weather in Pune?")])

if ai_msg.tool_calls:
    print(get_weather.invoke(ai_msg.tool_calls[0]["args"]))
else:
    print(ai_msg.content)