# Contador de Moedas – Processamento de Imagens

Este projeto utiliza **Python**, **OpenCV** e **scikit-learn** para detectar moedas em uma imagem, classificá-las e retornar seus valores.  
Ele também possui um pequeno servidor Flask que permite enviar imagens e receber o resultado do contador.

---

## 📌 Objetivo do Projeto

- Detectar moedas em uma imagem usando **técnicas de processamento de imagem**  
- Classificar o valor das moedas com base em características visuais  
- Retornar a contagem total e o valor acumulado  
- Fornecer uma interface simples via API com **Flask**

O projeto foi desenvolvido para fins acadêmicos, mostrando na prática como visão computacional pode ser utilizada para reconhecimento de padrões.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **OpenCV**
- **NumPy**
- **scikit-learn**
- **Flask**
- **Pillow**

---

## 📥 Como Baixar e Rodar

### 1. Clonar o repositório
```bash
git clone <seu-link-aqui>
cd <nome-da-pasta>
```

### 2. Criar ambiente virtual
```bash
python -m venv .venv
```

### 3. Ativar o ambiente virtual

#### Windows:
```bash
.venv\Scripts\activate
```

#### Linux/Mac:
```bash
source .venv/bin/activate
```

### 4. Instalar dependências
```bash
pip install -r requirements.txt
```

### 5. Rodar o servidor Flask
```bash
python contador_server.py
```

---

## 🖼️ Estrutura do Projeto

```
├── contador_classificar.py     # Lógica de detecção e classificação
├── contador_server.py          # Servidor Flask
├── uploads/                    # Imagens enviadas (se aplicável)
├── results/                    # Resultados gerados
├── moedas.jpg                  # Exemplo de imagem usada no projeto
├── requirements.txt            # Dependências
└── .gitignore
```

---

## 💡 Observações

- A pasta `.venv/` não deve ser enviada ao GitHub.  
- Você pode substituir a imagem `moedas.jpg` pelas suas próprias imagens de moedas para novos testes.  
- O servidor Flask pode ser estendido para criar uma interface web, se desejado.

---

## 📚 Licença
Este projeto é de uso livre para fins educativos.

