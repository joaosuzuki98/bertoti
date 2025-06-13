import smtplib
from os import getenv

try:
    with smtplib.SMTP(getenv("SMTP_SERVER"), int(getenv("SMTP_PORT"))) as server:
        server.ehlo()
        server.starttls()
        server.login(getenv("EMAIL_USER"), getenv("EMAIL_PASSWORD"))
        print("Conexão SMTP bem-sucedida!")
except Exception as e:
    print(f"Falha na conexão SMTP: {e}")