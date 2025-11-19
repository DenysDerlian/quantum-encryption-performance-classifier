# Projeto: Classificação de Desempenho em Criptografia Quântica — Resultados

Este documento apresenta a análise detalhada da base de dados, o pré-processamento, a evolução dos modelos de redes neurais e a comparação com modelos clássicos de Machine Learning. O objetivo é classificar o desempenho da rede de criptografia quântica em "Optimal" ou "Suboptimal".

---

## 1. Base de Dados e Pré-processamento

**Fonte:** Notebook `1.AnaliseExploratoria-Preprocessamento-QuantumEncryption-DenysDerlian.ipynb`

### Caracterização da Base de Dados

O problema abordado é a classificação binária de desempenho em um ambiente de criptografia quântica.

- **Tamanho do Dataset:** 1.000 amostras totais.
- **Variável Alvo:** `Performance_Target` (Binária: 1 = "Optimal", 0 = "Suboptimal").
- **Desbalanceamento:** A base é altamente desbalanceada, com aproximadamente **92%** (920 amostras) da classe negativa ("Suboptimal") e apenas **8%** (80 amostras) da classe positiva ("Optimal").
  - **Implicação:** Modelos ingênuos que sempre predizem "Suboptimal" alcançariam 92% de acurácia, mas 0% de Recall para a classe de interesse, tornando essa métrica inadequada como critério principal de avaliação.
- **Variáveis (Features):** O dataset é heterogêneo, contendo:
  - **Numéricas Contínuas:** Ex: `Throughput_Mbps`, `Latency_ms`, `Energy_Consumption_kWh`, métricas de uso de CPU/Memória.
  - **Categóricas:** Ex: `Attack_Type`, `Vulnerability_Level`, `Optimization_Level`, `Quantum_Protocol_Type`.
  - **Numéricas Discretas (tratadas como categóricas):** `Quantum_Key_Size_bits`, `Real_Time_Processing`.

### Tratamento e Pré-processamento

Para lidar com a heterogeneidade e a qualidade dos dados, as seguintes estratégias foram adotadas:

1. **Tratamento de Valores Ausentes:**
   - Variáveis categóricas com dados faltantes (`Attack_Type`, `Vulnerability_Level`, `Optimization_Level`) foram preenchidas com a categoria **"Unknown"**. Essa escolha preserva a informação de ausência, que pode ser sistemicamente relevante (ex: um ataque não identificado).

2. **Feature Engineering (Criação de Variáveis):**
   - Foram criadas novas features para capturar relações físicas e lógicas do domínio:
     - `Security_Index = Intrusion_Detection_Accuracy - Attack_Success_Rate`: Métrica composta que quantifica a margem de segurança efetiva do sistema.
     - `Latency_Efficiency = Throughput / (Encryption_Latency + 1)`: Razão que mede a eficiência do throughput em relação ao overhead de latência.
     - `Energy_per_Throughput = Energy_Consumption / (Throughput + ε)`: Indicador de eficiência energética (menor é melhor).
     - `Latency_mean`: Média agregada de múltiplas métricas de latência (Encryption, Decryption, Latency_ms) para reduzir dimensionalidade e capturar comportamento geral.

3. **Normalização e Codificação:**
   - **Numéricas:** Normalização **Min-Max Scaling** para o intervalo [0, 1]. Importante: O `fit` do scaler foi realizado dentro de cada fold da validação cruzada para evitar *data leakage*.
   - **Categóricas:** Codificação via **Label Encoding** para modelos de árvore e embeddings em redes neurais, ou **One-Hot Encoding** dependendo da necessidade do modelo.

4. **Estratégia para Desbalanceamento:**
   - **Métrica de Avaliação:** O foco principal foi a **PR-AUC (Area Under the Precision-Recall Curve)**, que é mais informativa que a ROC-AUC em cenários de classes raras.
   - **Validação:** Uso de **Stratified K-Fold Cross-Validation** para garantir que a proporção de classes se mantenha em todos os treinos e testes.
   - **Função de Perda (Redes Neurais):** Adoção da **Focal Loss**, que penaliza mais os erros em exemplos difíceis (geralmente da classe minoritária), controlada pelos hiperparâmetros `alpha` (balanceamento) e `gamma` (foco).

---

## 2. Evolução dos Modelos de Rede Neural (Otimizados)

O desenvolvimento das redes neurais seguiu uma abordagem incremental de complexidade para tentar capturar melhor as nuances dos dados.

### 2.1. Baseline MLP (Notebook 2)

- **Arquitetura:** Uma rede Perceptron Multicamadas (MLP) simples, totalmente conectada com 2 camadas ocultas (16 e 128 neurônios).
- **Entrada:** Todas as features (28 dimensões: 20 numéricas + 10 categóricas codificadas) concatenadas em um único vetor.
- **Ativação:** SELU (Self-Normalizing) para estabilização do gradiente.
- **Limitação:** Trata todas as variáveis de forma homogênea, dificultando a extração de padrões específicos de variáveis categóricas versus numéricas.

**Espaço de Hiperparâmetros (Optuna - 100 trials):**

- `n_layers`: {2, 3, 4, 5}
- `n_units_l{i}`: {16, 32, 48, ..., 128} (step=16)
- `activation_l{i}`: {relu, tanh, elu, selu}
- `lr` (learning rate): [1×10⁻⁵, 1×10⁻²] (log scale)
- `l2` (regularização): [1×10⁻⁶, 1×10⁻³] (log scale)
- `dropout`: [0.1, 0.5]
- `batch_size`: {16, 32, 64}

### 2.2. MLP Paralelo (Notebook 3)

- **Arquitetura:** Rede com dois ramos (branches) especializados:
  - *Ramo Numérico (20 features):* 3 camadas densas (224→128→160 neurônios) processando variáveis contínuas.
  - *Ramo Categórico (10 features):* 1 camada densa (48 neurônios) processando embeddings categóricos com cardinalidades de 3-5 valores por variável.
- **Fusão:** Concatenação dos outputs dos ramos seguida por 1 camada densa (192 neurônios).
- **Ativação:** ELU (Exponential Linear Unit).
- **Vantagem:** Permite que a rede aprenda representações especializadas para cada tipo de dado antes de combiná-las, evitando compromissos de representação.

**Espaço de Hiperparâmetros (Optuna - 100 trials):**

- `n_numeric_layers`: {2, 3, 4}
- `numeric_units_l{i}`: {32, 64, 96, ..., 256} (step=32)
- `n_categorical_layers`: {1, 2, 3}
- `categorical_units_l{i}`: {16, 32, 48, ..., 128} (step=16)
- `n_merged_layers`: {1, 2, 3}
- `merged_units_l{i}`: {32, 64, 96, ..., 256} (step=32)
- `activation`: {relu, elu, selu}
- `lr`: [1×10⁻⁵, 1×10⁻²] (log scale)
- `l2`: [1×10⁻⁶, 1×10⁻³] (log scale)
- `dropout`: [0.1, 0.5]
- `batch_size`: {16, 32, 64}
- `focal_gamma`: [0.3, 3.0]

### 2.3. Attention Multi-Headed MLP (Notebook 4)

- **Arquitetura:** A mais sofisticada, dividindo as features em **4 ramos especializados**:
  - *Numérico Positivo (9 features):* Variáveis com correlação positiva ao alvo (ex: Throughput, CPU_Utilization).
  - *Numérico Negativo (9 features):* Variáveis com correlação negativa (ex: Latency, Energy_Consumption).
  - *Categórico Positivo (3 features):* Categorias correlacionadas positivamente (ex: Optimization_Level).
  - *Categórico Negativo (7 features):* Categorias correlacionadas negativamente (ex: Vulnerability_Level).
- **Mecanismo de Atenção:** Aplica camadas de atenção "soft" (48 unidades) dentro de cada ramo para ponderar dinamicamente a importância de cada feature.
- **Fusão:** Concatenação dos outputs atencionais seguida por 2 camadas densas (64→192 neurônios).
- **Ativação:** SELU.
- **Vantagem:** O modelo foca nas features mais relevantes para cada exemplo, capturando interações específicas intra-grupo e explorando a estrutura de correlação com o alvo.

**Espaço de Hiperparâmetros (Optuna - 100 trials):**

- `n_numeric_layers`: {2, 3, 4} (por ramo)
- `numeric_units_l{i}`: {32, 48, 64, ..., 128} (step=16)
- `n_categorical_layers`: {1, 2, 3} (por ramo)
- `categorical_units_l{i}`: {16, 32, 48, 64} (step=16)
- `attention_units`: {16, 32, 48, 64} (step=16)
- `n_merged_layers`: {1, 2, 3}
- `merged_units_l{i}`: {64, 96, 128, ..., 256} (step=32)
- `activation`: {relu, elu, selu}
- `lr`: [1×10⁻⁵, 1×10⁻²] (log scale)
- `l2`: [1×10⁻⁶, 1×10⁻³] (log scale)
- `dropout`: [0.1, 0.5]
- `batch_size`: {16, 32, 64}
- `focal_gamma`: [0.3, 3.0]

### Resultados das Redes Neurais (Média CV 5-Folds)

A tabela abaixo resume o desempenho dos modelos otimizados.

| Modelo            |   PR-AUC   |  ROC-AUC   |  F1-Score  | Precision  |   Recall   |  Accuracy  |
| :---------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| **Baseline MLP**  |   0.7974   |   0.9699   |   0.6547   |   0.5488   |   0.8169   |   0.9290   |
| **MLP Paralelo**  |   0.9354   |   0.9919   |   0.7926   |   0.7432   |   0.8765   |   0.9620   |
| **Attention MLP** | **0.9721** | **0.9971** | **0.8413** | **0.7435** | **0.9757** | **0.9690** |

**Análise:**

- **Evolução Incremental:** Houve ganhos consistentes a cada incremento de complexidade:
  - Baseline → Paralelo: +0.138 pontos em PR-AUC (+17.3% relativo), validando a separação de ramos especializados.
  - Paralelo → Atenção: +0.037 pontos em PR-AUC (+4.0% relativo), demonstrando benefício da atenção contextual.
- **Destaque do Modelo Attention MLP:**
  - Alcançou **PR-AUC = 0.9721** e **Recall = 97.57%** (melhor entre todos os modelos de deep learning).
  - Apenas 2.43% das instâncias "Optimal" foram classificadas incorretamente como "Suboptimal" (Falsos Negativos).
  - Trade-off consciente: Precision de 74.35% implica ~25% de Falsos Positivos, mas maximiza a detecção da classe crítica.
- **Efetividade da Focal Loss:** A função de perda com `gamma=0.63` permitiu foco em exemplos difíceis, melhorando o Recall sem colapsar totalmente a Precision.

---

## 3. Modelos Clássicos (Notebook 5)

Para comparação, foram treinados e otimizados (via Optuna) três modelos clássicos robustos: **Random Forest (RF)**, **Support Vector Machine (SVM)** e **XGBoost**.

### Resultados dos Modelos Clássicos (Média CV 5-Folds)

| Modelo            |   PR-AUC   |  ROC-AUC   |  F1-Score  | Precision  | Recall |  Accuracy  |
| :---------------- | :--------: | :--------: | :--------: | :--------: | :----: | :--------: |
| **Random Forest** | **1.0000** | **1.0000** | **0.9579** | **1.0000** | 0.9231 | **0.9938** |
| **XGBoost**       |   0.9855   |   0.9975   |   0.9579   | **1.0000** | 0.9231 | **0.9938** |
| **SVM**           |   0.7672   |   0.9554   |   0.5519   |   0.8133   | 0.4418 |   0.9425   |

**Análise:**

- **Random Forest e XGBoost — Performance Excepcional:**
  - RF alcançou métricas quase perfeitas (PR-AUC ≈ 1.00, Precision = 1.00, Recall = 0.92), com matriz de confusão acumulada mostrando apenas 5 Falsos Negativos em 800 amostras de teste.
  - XGBoost ficou muito próximo (PR-AUC = 0.9855), com desempenho idêntico ao RF em F1-Score e Recall.
  - **Razão do sucesso:** Árvores de decisão lidam nativamente com features heterogêneas, criam fronteiras não-lineares complexas e, com `class_weight` (RF) e `scale_pos_weight` (XGBoost), priorizam corretamente a classe minoritária.

**Matriz de Confusão — Random Forest (CV 5-Folds acumulada):**

```text
              Predito
              Neg    Pos
Real  Neg  [[ 734      0 ]
Real  Pos   [   5     61 ]]
```

- **Verdadeiros Negativos (TN):** 734
- **Falsos Positivos (FP):** 0
- **Falsos Negativos (FN):** 5 (7.7% da classe positiva)
- **Verdadeiros Positivos (TP):** 61

**Espaço de Hiperparâmetros — Random Forest (Optuna - 100 trials):**

- `n_estimators`: {100, 150, 200, ..., 500} (step=50)
- `max_depth`: {10, 15, 20, ..., 50} (step=5)
- `min_samples_split`: {2, 3, ..., 20}
- `min_samples_leaf`: {1, 2, ..., 10}
- `max_features`: {sqrt, log2, 0.5, 0.7}
- `class_weight`: balanced (automático)

**Matriz de Confusão — XGBoost (CV 5-Folds acumulada):**

```text
              Predito
              Neg    Pos
Real  Neg  [[ 734      0 ]
Real  Pos   [   5     61 ]]
```

- **Verdadeiros Negativos (TN):** 734
- **Falsos Positivos (FP):** 0
- **Falsos Negativos (FN):** 5 (7.7% da classe positiva)
- **Verdadeiros Positivos (TP):** 61

**Espaço de Hiperparâmetros — XGBoost (Optuna - 100 trials):**

- `n_estimators`: {100, 150, 200, ..., 500} (step=50)
- `max_depth`: {3, 4, ..., 10}
- `learning_rate`: [0.01, 0.3] (log scale)
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `min_child_weight`: {1, 2, ..., 10}
- `gamma`: [0.0, 5.0]
- `reg_alpha` (L1): [0.0, 10.0]
- `reg_lambda` (L2): [0.0, 10.0]
- `scale_pos_weight`: 11.5 (razão negativo/positivo)

- **SVM — Limitações Evidentes:**
  - Pior desempenho geral (PR-AUC = 0.7672, Recall = 0.44), com matriz confusa acumulada mostrando 37 Falsos Negativos (56% da classe positiva).
  - Apesar de alta Precision (0.81), o modelo falha em detectar mais da metade dos casos "Optimal".
  - **Limitação conceitual:** Mesmo com kernel RBF, a fronteira de decisão do SVM teve dificuldade em separar a classe minoritária em um espaço de alta dimensionalidade com interações complexas.

**Matriz de Confusão — SVM (CV 5-Folds acumulada):**

```text
              Predito
              Neg    Pos
Real  Neg  [[ 725      9 ]
Real  Pos   [  37     29 ]]
```

- **Verdadeiros Negativos (TN):** 725
- **Falsos Positivos (FP):** 9 (1.2% da classe negativa)
- **Falsos Negativos (FN):** 37 (56.1% da classe positiva) ⚠️
- **Verdadeiros Positivos (TP):** 29

**Espaço de Hiperparâmetros — SVM (Optuna - 100 trials):**

- `C` (regularização): [0.01, 100] (log scale)
- `kernel`: {rbf, poly, sigmoid}
- `degree`: {2, 3, 4, 5} (se kernel='poly')
- `gamma`: {scale, auto}
- `class_weight`: balanced (automático)

---

## 4. Ranking Geral e Discussão Final

Abaixo, o ranking de todos os modelos desenvolvidos, ordenados pela métrica principal **PR-AUC**.

| Ranking | Modelo            |   PR-AUC   |  F1-Score  |   Recall   | Precision  |
| :-----: | :---------------- | :--------: | :--------: | :--------: | :--------: |
| **1º**  | **Random Forest** | **1.0000** | **0.9579** |   0.9231   | **1.0000** |
| **2º**  | **XGBoost**       |   0.9855   | **0.9579** |   0.9231   | **1.0000** |
| **3º**  | **Attention MLP** |   0.9721   |   0.8413   | **0.9757** |   0.7435   |
| **4º**  | **MLP Paralelo**  |   0.9354   |   0.7926   |   0.8765   |   0.7432   |
| **5º**  | **Baseline MLP**  |   0.7974   |   0.6547   |   0.8169   |   0.5488   |
| **6º**  | **SVM**           |   0.7672   |   0.5519   |   0.4418   |   0.8133   |

### Discussão dos Resultados

1. **Supremacia dos Modelos Baseados em Árvore (RF e XGBoost):**
   - Os modelos de árvore (Random Forest e XGBoost) dominaram o ranking. Isso é comum em dados tabulares heterogêneos, onde árvores conseguem criar fronteiras de decisão "retangulares" precisas e lidar nativamente com interações não-lineares entre variáveis categóricas e numéricas sem a necessidade de arquiteturas complexas.
   - A precisão perfeita (1.0) indica que esses modelos são extremamente confiáveis: quando predizem "Optimal", estão certos.

2. **Desempenho Competitivo da Rede Neural com Atenção:**
   - O modelo **Attention MLP** (3º lugar) ficou muito próximo dos líderes em PR-AUC (0.97 vs 0.98/1.00).
   - **Destaque para o Recall:** A rede neural com atenção obteve o **maior Recall de todos (97.57%)**, superando até o Random Forest (92.31%). Isso significa que, se o objetivo crítico for *não perder nenhum caso Optimal* (evitar Falsos Negativos), a Rede Neural com Atenção seria a melhor escolha, mesmo com uma precisão ligeiramente menor.
   - A evolução (Baseline -> Paralelo -> Atenção) prova que arquiteturas de Deep Learning para dados tabulares precisam de design específico (atenção, ramos) para competir com Gradient Boosting.

3. **Dificuldades do SVM e Baseline MLP:**
   - **Baseline MLP:** Sofreu com a representação homogênea de features heterogêneas, resultando em Precision baixa (0.55) — para cada 2 predições "Optimal", uma estava errada (alto custo de Falsos Positivos).
   - **SVM:** Apesar de Precision razoável (0.81), o Recall crítico de apenas 44% indica que o modelo classifica erroneamente mais da metade dos casos positivos como negativos, falhando no objetivo principal.

### Implicações Práticas

- **Para ambientes de produção:** Random Forest ou XGBoost são recomendados pela confiabilidade (Precision perfeita) e balanceamento geral.
- **Para sistemas críticos de segurança:** O Attention MLP pode ser preferível quando o custo de um Falso Negativo (não detectar desempenho "Optimal") é muito maior que o de um Falso Positivo.
- **Trade-off fundamental:** Modelos baseados em árvore minimizam Falsos Positivos (Precision = 1.0); o Attention MLP minimiza Falsos Negativos (Recall = 0.98).

### Limitações e Trabalhos Futuros

1. **Tamanho do Dataset:** Com apenas 1.000 amostras (80 positivas), há risco de overfitting, especialmente nos modelos de árvore que mostram métricas quase perfeitas. Validação em dataset independente é essencial.
2. **Interpretabilidade:** Modelos de árvore oferecem importância de features clara; redes neurais com atenção são mais opacas, dificultando análise de erros.
3. **Custo Computacional:** O Attention MLP requer mais recursos para treinamento (otimização Optuna com 100+ trials) comparado a Random Forest.
4. **Generalização:** Avaliação em dados de diferentes ambientes de criptografia quântica ajudaria a validar a robustez dos modelos.

**Conclusão Final:** Para o problema de classificação de desempenho em Criptografia Quântica, o **Random Forest** é o modelo mais equilibrado e robusto, oferecendo confiabilidade máxima (Precision = 1.0) com excelente Recall (92%). No entanto, o **Attention MLP** demonstrou ser uma alternativa competitiva e poderosa, especialmente quando a prioridade é maximizar a detecção da classe "Optimal" (Recall = 98%), justificando sua complexidade arquitetural em cenários onde Falsos Negativos têm custo crítico.

