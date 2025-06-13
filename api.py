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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Carregar variáveis de ambiente
load_dotenv()
HG_TOKEN = os.getenv("HG_TOKEN")



# Carregar variáveis de e-mail com verificações
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))  # Conversão segura com valor padrão

# Verificação imediata
if None in [EMAIL_USER, EMAIL_PASSWORD, SMTP_SERVER]:
    raise ValueError("Variáveis de e-mail essenciais não encontradas no .env")

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

def send_email(to_email, subject, body, attachment=None):
    try:
        # Verificação adicional das variáveis
        if None in [EMAIL_USER, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT]:
            raise ValueError("Variáveis de e-mail não configuradas corretamente no .env")

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if attachment:
            attachment.seek(0)  # Reset file pointer
            part = MIMEApplication(
                attachment.read(),
                Name=attachment.filename
            )
            part['Content-Disposition'] = f'attachment; filename="{attachment.filename}"'
            msg.attach(part)

        # Conexão mais robusta com tratamento de timeout
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"E-mail enviado com sucesso para {to_email}")
            return True

    except smtplib.SMTPAuthenticationError:
        print("Erro de autenticação: Verifique EMAIL_USER e EMAIL_PASSWORD no .env")
        return False
    except Exception as e:
        print(f"Erro ao enviar e-mail: {str(e)}")
        return False

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
        prompt = f"Este texto foi extraído de uma imagem: {extracted_text}. Vocẽ é um corretor de imóveis e deverá fazer uma avaliação do imóvel do texto extraído, diga pontos interessantes perto do imóvel e também preço médio dele, como o preço da metragem e afins. Responda em português brasileiro."
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

@app.route('/test-smtp-connection')
def test_smtp_connection():
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            return jsonify({
                "status": "success",
                "message": "Conexão SMTP bem-sucedida!",
                "server": SMTP_SERVER,
                "port": SMTP_PORT
            }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "loaded_vars": {
                "EMAIL_USER": EMAIL_USER,
                "SMTP_SERVER": SMTP_SERVER,
                "SMTP_PORT": SMTP_PORT
            }
        }), 500

@app.route('/send-email', methods=['POST'])
def send_email_route():
    if 'email' not in request.form:
        return jsonify({'error': 'Nenhum e-mail fornecido.'}), 400
    
    email = request.form['email']
    text = request.form.get('text', '')
    image = request.files.get('image')

    if not text and not image:
        return jsonify({'error': 'Nenhum conteúdo para enviar.'}), 400

    try:
        subject = "Resultado da extração de texto da imagem"
        body = f"Segue o resultado da extração de texto:\n\n{text}\n\nAtenciosamente,\nSuzukAI"

        success = send_email(
            to_email=email,
            subject=subject,
            body=body,
            attachment=image
        )

        if success:
            return jsonify({'message': 'E-mail enviado com sucesso!'}), 200
        else:
            return jsonify({'error': 'Falha ao enviar e-mail.'}), 500

    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro ao enviar o e-mail: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)