from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import os
import datetime
import pytz
import yaml
from smolagents import CodeAgent, DuckDuckGoSearchTool, load_tool, tool, LiteLLMModel, HfApiModel
from tools.final_answer import FinalAnswerTool
from dotenv import load_dotenv
from flask_cors import CORS

# Carregar variáveis de ambiente
load_dotenv()
HG_TOKEN = os.getenv("HG_TOKEN")

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas as rotas

# Função para extrair texto de uma imagem usando OCR
def extract_text_from_image(image_stream) -> str:
    try:
        image = Image.open(image_stream)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Erro ao extrair texto: {str(e)}"

# Carregar modelos e ferramentas
final_answer = FinalAnswerTool()

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
    token=HG_TOKEN
)

lite_model = LiteLLMModel(
    model_id="ollama/deepseek-r1:7b",
    temperature=0.6, 
    max_tokens=200
)

# Importar ferramenta de geração de imagem
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Carregar templates de prompt
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), image_generation_tool],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhum arquivo de imagem fornecido.'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'Nome de arquivo inválido.'}), 400

    try:
        # Extrair texto da imagem
        extracted_text = extract_text_from_image(image_file.stream)

        if not extracted_text:
            return jsonify({'error': 'Nenhum texto foi extraído da imagem.'}), 400

        # Analisar o texto usando o agente
        prompt = f"Este texto foi extraído de uma imagem: {extracted_text}. O que isso significa? Responda em português brasileiro."
        response = agent.run(prompt)

        return jsonify({'explanation': response}), 200

    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro ao processar a imagem: {str(e)}'}), 500
    
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhum arquivo de imagem fornecido.'}), 400

    image_file = request.files['image']
    user_prompt = request.form.get('prompt', '')  # Prompt opcional do usuário

    if image_file.filename == '':
        return jsonify({'error': 'Nome de arquivo inválido.'}), 400

    try:
        # Extrair texto da imagem
        extracted_text = extract_text_from_image(image_file.stream)

        if not extracted_text:
            return jsonify({'result': ''}), 200  # Sem texto extraído

        # Analisar com o agente apenas se houver prompt
        if user_prompt.strip():
            full_prompt = f"Texto extraído da imagem: {extracted_text}\n\nPergunta do usuário: {user_prompt}"
            response = agent.run(full_prompt)
            result = response
        else:
            result = extracted_text  # Apenas o texto extraído

        return jsonify({'result': result}), 200

    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro ao processar a imagem: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
