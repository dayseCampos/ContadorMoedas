# 🪙 Contador de Moedas – Visão Computacional com Python

Bem-vindo(a)!  
Este projeto utiliza **Python**, **OpenCV** e **scikit-learn** para detectar e classificar moedas em imagens, retornando tanto a quantidade quanto o valor total.  
Além disso, há um servidor **Flask**, permitindo enviar imagens e receber o resultado em formato JSON.

---

## ✨ Objetivo do Projeto

Este projeto foi desenvolvido para demonstrar, de forma prática e didática, como técnicas de **Processamento de Imagens** podem ser aplicadas para:

- 📸 Detectar moedas em uma imagem  
- 🔍 Extrair características relevantes  
- 🧠 Classificar o valor de cada moeda  
- 🧮 Somar automaticamente o valor total  
- 🌐 Servir tudo isso via API com Flask  

Ele foi usado como trabalho acadêmico, mostrando como visão computacional pode resolver problemas reais.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **OpenCV (cv2)**
- **NumPy**
- **scikit-learn** (KMeans)
- **Flask**
- **Pillow**

---

## 📥 Como Clonar e Rodar o Projeto

### 1️⃣ Clonar o repositório

```bash
git clone <seu-link-aqui>
cd <nome-da-pasta>
```

### 2️⃣ Criar ambiente virtual

```bash
python -m venv .venv
```

### 3️⃣ Ativar o ambiente virtual

#### Windows:
```bash
.venv\Scripts\activate
```

#### Linux/Mac:
```bash
source .venv/bin/activate
```

### 4️⃣ Instalar as dependências

```bash
pip install -r requirements.txt
```

### 5️⃣ Iniciar o servidor Flask

```bash
python contador_server.py
```

A API ficará disponível em:

```
http://localhost:5000
```

---

## 📂 Estrutura do Projeto

```
├── contador_classificar.py     # Lógica de detecção e classificação
├── contador_server.py          # Servidor Flask para a API
├── uploads/                    # Imagens enviadas (opcional)
├── results/                    # Resultados gerados
├── moedas.jpg                  # Imagem de exemplo
├── requirements.txt            # Dependências do projeto
└── .gitignore                  # Arquivos/pastas ignoradas
```

---

## 🧡 Licença

Este projeto é livre para uso acadêmico e educacional.
