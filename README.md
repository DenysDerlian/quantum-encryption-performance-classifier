# Quantum Encryption Performance Analysis

Este projeto apresenta uma análise completa de desempenho de criptografia quântica, incluindo análise exploratória, pré-processamento de dados e implementação de diversos modelos de machine learning.

## 📁 Estrutura do Projeto

```
.
├── 1.AnaliseExploratoria-Preprocessamento-QuantumEncryption-DenysDerlian.ipynb
├── 2.ModeloBaseline-MLP-QuantumEncryption-DenysDerlian.ipynb
├── 3.ModeloMLPParalelo-QuantumEncryption-DenysDerlian.ipynb
├── 4.ModeloAttentionMultiHeaded-QuantumEncryption-DenysDerlian.ipynb
├── 5.RF_SVM_XGBoost-QuantumEncryption-DenysDerlian.ipynb
├── data/
│   └── preprocessed/
│       ├── df_preprocessed.csv
│       ├── X_categorical_preprocessed.csv
│       ├── X_numerical_preprocessed.csv
│       └── y_preprocessed.csv
└── modules/
    └── config.py
```

## 📊 Dataset

Este projeto utiliza o **Quantum Encryption Performance Dataset** disponibilizado por Ziya no Kaggle.

**Créditos do Dataset:**
- **Autor:** Ziya
- **Fonte:** [Kaggle - Quantum Encryption Performance Dataset](https://www.kaggle.com/datasets/ziya07/quantum-encryption-performance-dataset)
- **Licença:** CC0: Public Domain
- **Descrição:** Dataset com 1000 amostras e 25 colunas cobrindo parâmetros de criptografia quântica, métricas de performance de rede, avaliação de segurança e características de big data.

### Sobre o Dataset

O dataset inclui dados sobre várias métricas relacionadas à criptografia quântica em contextos de segurança de rede, especialmente sob condições de big data. Principais categorias de dados:

- **Parâmetros de Criptografia Quântica:** Tamanho de chave, tipo de protocolo, métodos de distribuição, latências
- **Métricas de Performance de Rede:** Throughput, latência, perda de pacotes, utilização de banda
- **Métricas de Avaliação de Segurança:** Força de criptografia, tipos de ataque, níveis de vulnerabilidade
- **Características de Big Data:** Volume, variedade, velocidade e complexidade dos dados
- **Métricas de Recursos:** Utilização de CPU/memória, consumo de energia
- **Target:** Performance_Target (classificação binária: "Optimal" ou "Suboptimal")

## 🚀 Notebooks

1. **Análise Exploratória e Pré-processamento:** Exploração inicial dos dados e preparação para modelagem
2. **Modelo Baseline (MLP):** Implementação de rede neural Multi-Layer Perceptron básica
3. **Modelo MLP Paralelo:** Versão otimizada do MLP com processamento paralelo
4. **Modelo Attention Multi-Headed:** Implementação de arquitetura com mecanismo de atenção
5. **Random Forest, SVM e XGBoost:** Modelos de machine learning clássicos

## 🛠️ Requisitos

### Instalação

1. **Python 3.9+** é recomendado

2. **Instalar dependências:**

```bash
pip install -r requirements.txt
```

### Principais Bibliotecas

- **Data Science:** pandas, numpy, scipy
- **Machine Learning:** scikit-learn, xgboost, imbalanced-learn
- **Deep Learning:** TensorFlow 2.15+
- **Otimização:** Optuna
- **Visualização:** matplotlib, seaborn, plotly
- **Dataset:** kagglehub

### Configuração GPU (Opcional)

Para acelerar o treinamento das redes neurais, configure TensorFlow com suporte GPU:

```bash
# Para NVIDIA GPUs
pip install tensorflow[and-cuda]
```

## 👤 Autor

**Denys Derlian**

## 📄 Licença

Este projeto de análise está disponível para fins educacionais. O dataset original está sob licença CC0: Public Domain.
