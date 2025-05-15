from smolagents import CodeAgent,DuckDuckGoSearchTool,load_tool,tool,LiteLLMModel, HfApiModel
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
import ollama
import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from flask import Flask, jsonify, request

from Gradio_UI import GradioUI
HG_TOKEN = os.getenv("HG_TOKEN")

app = Flask(__name__)

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def extract_text_from_image(image_path: str) -> str:
    """Extracts text from an image using OCR.
    Args:
        image_path: Path to the image file.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return f"Extracted text: {text.strip()}"
    except Exception as e:
        return f"Error extracting text: {str(e)}"

final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
    custom_role_conversions=None,
    token=HG_TOKEN
)

# ollama_model = ollama.chat(model="deepseek-r1:7b")

lite_model = LiteLLMModel(
    model_id="ollama/deepseek-r1:7b",
    temperature=0.6, 
    max_tokens=200
)

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), image_generation_tool], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# Launches the GUI interface, just remove the comment from the line above
# GradioUI(agent).launch()
image_text = extract_text_from_image("./img/teste.png")
# user_input = input("Digite sua pergunta: ")
response = agent.run(f"this text got extracted from an image: {image_text}. What does that means? Answer in brazilian portuguese")
print(response)
