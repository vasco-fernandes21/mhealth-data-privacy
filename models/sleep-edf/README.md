# Sleep-EDF Baseline Model - Training Report

## 📊 Overview

Este documento descreve o processo de treino e os resultados obtidos para o modelo baseline LSTM no dataset Sleep-EDF Expanded.

## 🎯 Dataset Information

- **Dataset**: Sleep-EDF Expanded
- **Total de épocas processadas**: 453,005
- **Features por época**: 24 (8 features × 3 canais EEG/EOG)
- **Classes**: 5 (W, N1, N2, N3, R)
- **Divisão dos dados**:
  - Train: 317,103 épocas (70%)
  - Validation: 67,951 épocas (15%)
  - Test: 67,951 épocas (15%)

## 🏗️ Model Architecture

### LSTM Baseline Model
- **Tipo**: LSTM (Long Short-Term Memory)
- **Arquitetura**:
  - LSTM Layer: 128 unidades
  - Dropout: 0.3
  - Dense Layer: 64 unidades
  - Dropout: 0.3
  - Output Layer: 5 classes (softmax)

### Hyperparameters
- **Window Size**: 10 timesteps
- **Learning Rate**: 0.001 (inicial)
- **Batch Size**: 64
- **Epochs**: 100 (máximo)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## 📈 Training Process

### Training Configuration
- **Early Stopping**: Patience de 15 épocas
- **Learning Rate Reduction**: Factor 0.5, patience 5 épocas
- **Minimum Learning Rate**: 1e-6
- **Validation Monitoring**: val_loss

### Training Progress
- **Total Epochs Trained**: 36
- **Best Epoch**: 21
- **Early Stopping Triggered**: Sim (época 36)
- **Learning Rate Reduction**: 2 vezes
- **Final Learning Rate**: 6.25e-05

### Training Metrics Evolution

#### Loss Evolution
- **Initial Training Loss**: 0.544
- **Final Training Loss**: 0.336
- **Best Training Loss**: 0.336 (época 36)
- **Initial Validation Loss**: 0.448
- **Final Validation Loss**: 0.443
- **Best Validation Loss**: 0.361 (época 21)

#### Accuracy Evolution
- **Initial Training Accuracy**: 80.93%
- **Final Training Accuracy**: 87.88%
- **Best Training Accuracy**: 87.88% (época 36)
- **Initial Validation Accuracy**: 84.65%
- **Final Validation Accuracy**: 87.55%
- **Best Validation Accuracy**: 87.57% (época 21)

## 🎯 Final Results

### Test Set Performance
- **Accuracy**: 87.45%
- **Precision**: 85.72%
- **Recall**: 87.45%
- **F1-Score**: 85.82%

### Confusion Matrix
```
                Predicted
Actual    W     N1    N2    N3    R
W      42637   118   503    39   281
N1      1211   304  1572    17   591
N2       492   169 11217   420   658
N3        73     0   683  2012     3
R        423   107  1135    30  3247
```

### Class-wise Performance Analysis

#### Wake (W) - Classe 0
- **True Positives**: 42,637
- **False Positives**: 1,211 + 492 + 73 + 423 = 2,199
- **False Negatives**: 118 + 503 + 39 + 281 = 941
- **Precision**: 95.1%
- **Recall**: 97.8%

#### N1 Sleep - Classe 1
- **True Positives**: 304
- **False Positives**: 118 + 169 + 0 + 107 = 394
- **False Negatives**: 1,211 + 1,572 + 17 + 591 = 3,391
- **Precision**: 43.6%
- **Recall**: 8.2%

#### N2 Sleep - Classe 2
- **True Positives**: 11,217
- **False Positives**: 503 + 1,572 + 683 + 1,135 = 3,893
- **False Negatives**: 169 + 420 + 658 = 1,247
- **Precision**: 74.2%
- **Recall**: 90.0%

#### N3 Sleep - Classe 3
- **True Positives**: 2,012
- **False Positives**: 39 + 17 + 420 + 30 = 506
- **False Negatives**: 0 + 683 + 3 = 686
- **Precision**: 79.9%
- **Recall**: 74.6%

#### REM Sleep (R) - Classe 4
- **True Positives**: 3,247
- **False Positives**: 281 + 591 + 658 + 3 = 1,533
- **False Negatives**: 107 + 1,135 + 30 = 1,272
- **Precision**: 67.9%
- **Recall**: 71.9%

## 📊 Performance Analysis

### Strengths
1. **High Overall Accuracy**: 87.45% é um resultado sólido para classificação de estágios de sono
2. **Good Wake Detection**: Excelente performance na detecção de vigília (97.8% recall)
3. **Strong N2 Classification**: Boa performance no estágio N2 (90.0% recall)
4. **Stable Training**: Treino convergiu de forma estável sem overfitting significativo

### Challenges
1. **N1 Sleep Difficulty**: Baixa performance na classificação de N1 (8.2% recall)
2. **Class Imbalance**: Desequilíbrio entre classes afeta performance
3. **N1/N2 Confusion**: Confusão frequente entre estágios N1 e N2

### Training Stability
- **Convergence**: Modelo convergiu na época 21
- **Overfitting Control**: Early stopping funcionou adequadamente
- **Learning Rate Adaptation**: Redução automática do learning rate melhorou estabilidade

## 💾 Saved Files

### Model Files
- **Model**: `lstm_baseline_sleep-edf.h5` (1.0MB)
- **Training History**: `history_baseline_sleep-edf.json` (3.5KB)
- **Results**: `results_baseline_sleep-edf.json` (501B)

### File Contents
- **Model**: Arquitetura e pesos treinados do modelo LSTM
- **History**: Métricas de treino e validação por época
- **Results**: Métricas finais de teste e matriz de confusão

## 🔄 Next Steps

1. **Differential Privacy Training**: Treinar modelo com DP para comparar trade-offs
2. **Federated Learning**: Implementar FL para preservação de privacidade
3. **Hyperparameter Tuning**: Otimizar parâmetros para melhorar performance em N1
4. **Data Augmentation**: Explorar técnicas para equilibrar classes
5. **Ensemble Methods**: Combinar múltiplos modelos para melhor performance

## 📝 Technical Notes

### Data Preprocessing
- Features extraídas: domínio temporal e frequencial
- Normalização: StandardScaler aplicado
- Window Size: 10 timesteps para LSTM
- Sampling: 100 Hz para sinais EEG/EOG

### Training Environment
- **Framework**: TensorFlow/Keras
- **Hardware**: CPU training
- **Memory Usage**: ~8GB RAM durante treino
- **Training Time**: ~46 segundos por época

### Model Complexity
- **Total Parameters**: ~50,000
- **Trainable Parameters**: ~50,000
- **Model Size**: 1.0MB

---

*Relatório gerado automaticamente após treino do modelo baseline LSTM no dataset Sleep-EDF Expanded.*
