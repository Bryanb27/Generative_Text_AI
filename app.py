#Imports
from transformers import AutoTokenizer, BertModel
import openai 
from PyPDF2 import PdfReader
import os
import torch
from flask import Flask, render_template, request

#Carregar modelo Bert
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#Chave openai
openai.api_key = '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' 

#Diretorio dos pdfs
diretorio_pdfs = 'PDFs'

#Extrar texto dos pdfs
def extrair_texto(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages[:2]:  #Por enquanto limitei a duas paginas (estava usando artigos para teste)
            text += page.extract_text()
        return text

#Gerar embeddings
def gerar_embeddings_pdfs():
    arquivos_pdfs = os.listdir(diretorio_pdfs)
    embeddings = []
    for arquivo in arquivos_pdfs:
        if arquivo.endswith('.pdf'):
            caminho_pdf = os.path.join(diretorio_pdfs, arquivo)
            texto_extraido = extrair_texto(caminho_pdf)
            
            #Bert para a geração dos embeddings
            inputs = tokenizer(texto_extraido, return_tensors='pt', max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    #print(embeddings)
    return embeddings


embeddings_indexados = gerar_embeddings_pdfs()

def limitar_tokens(embeddings_indexados):
    #Juntar todos os textos
    texto_completo = " ".join(" ".join(str(x) for x in embedding) for embedding in embeddings_indexados)
    
    #Dividir em tokens de 4097 pq é o limite do gpt-3.5turbo
    tokens = texto_completo.split()[:4097]
    
    #return contexto
    return tokens[0]

#Aplicacao em flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/responder_pergunta', methods=['POST'])
def responder_pergunta():
    #Receber a pergunta do formulario
    pergunta = request.form['pergunta']
    
    #Limite do gpt-3.5-turbo
    contexto = limitar_tokens(embeddings_indexados)
 
    resposta = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": "Você está respondendo a uma pergunta."},
        {"role": "user", "content": pergunta},
        {"role": "system", "content": contexto}
    ])
    resposta_texto = resposta['choices'][0]['message']['content']
    
    #Pagina resposta
    return render_template('resultado.html', pergunta=pergunta, resposta=resposta_texto)

if __name__ == '__main__':
    app.run()
