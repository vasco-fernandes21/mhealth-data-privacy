"""
Treino do Modelo LSTM Baseline (sem técnicas de privacidade)
Aplica-se a Sleep-EDF e WESAD após pré-processamento

Estrutura:
1. Reformatar dados para ser compatível com LSTM (séries temporais)
2. Treinar LSTM baseline
3. Avaliar em validação e teste
4. Guardar métricas e modelo para comparação posterior com DP/FL
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os

# Seed para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# PASSO 1: Carregar dados preprocessados
# ============================================================================

print("="*70)
print("TREINO LSTM BASELINE")
print("="*70)

# Sleep-EDF
print("\nCarregando Sleep-EDF...")
X_train_sleep = np.load('sleep_edf/X_train.npy')
X_val_sleep = np.load('sleep_edf/X_val.npy')
X_test_sleep = np.load('sleep_edf/X_test.npy')
y_train_sleep = np.load('sleep_edf/y_train.npy')
y_val_sleep = np.load('sleep_edf/y_val.npy')
y_test_sleep = np.load('sleep_edf/y_test.npy')

# WESAD
print("Carregando WESAD...")
X_train_wesad = np.load('wesad/X_train.npy')
X_val_wesad = np.load('wesad/X_val.npy')
X_test_wesad = np.load('wesad/X_test.npy')
y_train_wesad = np.load('wesad/y_train.npy')
y_val_wesad = np.load('wesad/y_val.npy')
y_test_wesad = np.load('wesad/y_test.npy')

print("Dados carregados com sucesso!")
print(f"Sleep-EDF - Train: {X_train_sleep.shape}, Val: {X_val_sleep.shape}, Test: {X_test_sleep.shape}")
print(f"WESAD - Train: {X_train_wesad.shape}, Val: {X_val_wesad.shape}, Test: {X_test_wesad.shape}")

# ============================================================================
# PASSO 2: Reformatar dados para LSTM
# ============================================================================
# LSTM espera: (n_samples, n_timesteps, n_features)
# Se os dados vêm já como (n_samples, n_features), precisas adicionar dimensão temporal

def reshape_for_lstm(X, window_size=10):
    """
    Reformata dados (n_samples, n_features) para (n_samples, window_size, n_features)
    
    Cria janelas temporais deslizantes:
    - window_size=10 significa que cada amostra é formada por 10 timesteps anteriores
    - Reduz o número de amostras para: n_samples - window_size + 1
    """
    n_samples, n_features = X.shape
    X_reshaped = []
    
    for i in range(n_samples - window_size + 1):
        X_reshaped.append(X[i:i+window_size, :])
    
    return np.array(X_reshaped)

# Aplicar reshape para LSTM
print("\nReformando dados para LSTM (window_size=10)...")
window_size = 10

X_train_sleep_lstm = reshape_for_lstm(X_train_sleep, window_size)
X_val_sleep_lstm = reshape_for_lstm(X_val_sleep, window_size)
X_test_sleep_lstm = reshape_for_lstm(X_test_sleep, window_size)

# Ajustar labels (remover primeiros window_size-1 labels)
y_train_sleep_lstm = y_train_sleep[window_size-1:]
y_val_sleep_lstm = y_val_sleep[window_size-1:]
y_test_sleep_lstm = y_test_sleep[window_size-1:]

X_train_wesad_lstm = reshape_for_lstm(X_train_wesad, window_size)
X_val_wesad_lstm = reshape_for_lstm(X_val_wesad, window_size)
X_test_wesad_lstm = reshape_for_lstm(X_test_wesad, window_size)

y_train_wesad_lstm = y_train_wesad[window_size-1:]
y_val_wesad_lstm = y_val_wesad[window_size-1:]
y_test_wesad_lstm = y_test_wesad[window_size-1:]

print(f"Sleep-EDF LSTM shapes - Train: {X_train_sleep_lstm.shape}, Val: {X_val_sleep_lstm.shape}, Test: {X_test_sleep_lstm.shape}")
print(f"WESAD LSTM shapes - Train: {X_train_wesad_lstm.shape}, Val: {X_val_wesad_lstm.shape}, Test: {X_test_wesad_lstm.shape}")

# Converter labels para categorical (one-hot encoding)
n_classes_sleep = len(np.unique(y_train_sleep_lstm))
n_classes_wesad = len(np.unique(y_train_wesad_lstm))

y_train_sleep_cat = to_categorical(y_train_sleep_lstm, n_classes_sleep)
y_val_sleep_cat = to_categorical(y_val_sleep_lstm, n_classes_sleep)
y_test_sleep_cat = to_categorical(y_test_sleep_lstm, n_classes_sleep)

y_train_wesad_cat = to_categorical(y_train_wesad_lstm, n_classes_wesad)
y_val_wesad_cat = to_categorical(y_val_wesad_lstm, n_classes_wesad)
y_test_wesad_cat = to_categorical(y_test_wesad_lstm, n_classes_wesad)

print(f"\nClasses - Sleep-EDF: {n_classes_sleep}, WESAD: {n_classes_wesad}")

# ============================================================================
# PASSO 3: Definir função de avaliação
# ============================================================================

def evaluate_model(model, X_val, X_test, y_val, y_test, dataset_name="Dataset"):
    """
    Avaliar modelo em validação e teste
    Retorna dicionário com métricas
    """
    # Predições
    y_val_pred_probs = model.predict(X_val, verbose=0)
    y_test_pred_probs = model.predict(X_test, verbose=0)
    
    # Converter para classe
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)
    
    # Labels originais (antes de one-hot)
    y_val_original = np.argmax(y_val, axis=1)
    y_test_original = np.argmax(y_test, axis=1)
    
    # Métricas de validação
    val_acc = accuracy_score(y_val_original, y_val_pred)
    val_prec = precision_score(y_val_original, y_val_pred, average='weighted', zero_division=0)
    val_rec = recall_score(y_val_original, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val_original, y_val_pred, average='weighted', zero_division=0)
    
    # Métricas de teste
    test_acc = accuracy_score(y_test_original, y_test_pred)
    test_prec = precision_score(y_test_original, y_test_pred, average='weighted', zero_division=0)
    test_rec = recall_score(y_test_original, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_original, y_test_pred, average='weighted', zero_division=0)
    
    results = {
        'dataset': dataset_name,
        'val_accuracy': float(val_acc),
        'val_precision': float(val_prec),
        'val_recall': float(val_rec),
        'val_f1': float(val_f1),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1)
    }
    
    print(f"\n{dataset_name} - LSTM Baseline Results:")
    print(f"  Validação - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
    print(f"  Teste     - Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")
    
    return results

# ============================================================================
# PASSO 4: Construir modelo LSTM
# ============================================================================

def build_lstm_model(input_shape, n_classes):
    """
    Arquitetura LSTM recomendada:
    Input → LSTM(64) → Dropout(0.5) → Dense(32) → Dropout(0.5) → Output
    
    Args:
        input_shape: (n_timesteps, n_features)
        n_classes: número de classes para classificação
    """
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# PASSO 5: Treinar LSTM - SLEEP-EDF
# ============================================================================

print("\n" + "="*70)
print("TREINO LSTM BASELINE - SLEEP-EDF")
print("="*70)

input_shape_sleep = (window_size, X_train_sleep_lstm.shape[2])
lstm_sleep = build_lstm_model(input_shape_sleep, n_classes_sleep)

print(f"\nArquitetura LSTM:")
lstm_sleep.summary()

print("\nTreinando LSTM (Sleep-EDF)...")
history_sleep = lstm_sleep.fit(
    X_train_sleep_lstm, y_train_sleep_cat,
    validation_data=(X_val_sleep_lstm, y_val_sleep_cat),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Avaliar Sleep-EDF
results_sleep = evaluate_model(
    lstm_sleep, X_val_sleep_lstm, X_test_sleep_lstm,
    y_val_sleep_cat, y_test_sleep_cat,
    dataset_name="Sleep-EDF"
)

# Guardar modelo e histórico
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

lstm_sleep.save('models/lstm_baseline_sleep_edf.h5')
with open('results/history_sleep_edf_baseline.json', 'w') as f:
    json.dump({
        'loss': history_sleep.history['loss'],
        'val_loss': history_sleep.history['val_loss'],
        'accuracy': history_sleep.history['accuracy'],
        'val_accuracy': history_sleep.history['val_accuracy']
    }, f)

# ============================================================================
# PASSO 6: Treinar LSTM - WESAD
# ============================================================================

print("\n" + "="*70)
print("TREINO LSTM BASELINE - WESAD")
print("="*70)

input_shape_wesad = (window_size, X_train_wesad_lstm.shape[2])
lstm_wesad = build_lstm_model(input_shape_wesad, n_classes_wesad)

print("\nTreinando LSTM (WESAD)...")
history_wesad = lstm_wesad.fit(
    X_train_wesad_lstm, y_train_wesad_cat,
    validation_data=(X_val_wesad_lstm, y_val_wesad_cat),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Avaliar WESAD
results_wesad = evaluate_model(
    lstm_wesad, X_val_wesad_lstm, X_test_wesad_lstm,
    y_val_wesad_cat, y_test_wesad_cat,
    dataset_name="WESAD"
)

# Guardar modelo e histórico
lstm_wesad.save('models/lstm_baseline_wesad.h5')
with open('results/history_wesad_baseline.json', 'w') as f:
    json.dump({
        'loss': history_wesad.history['loss'],
        'val_loss': history_wesad.history['val_loss'],
        'accuracy': history_wesad.history['accuracy'],
        'val_accuracy': history_wesad.history['val_accuracy']
    }, f)

# ============================================================================
# PASSO 7: Guardar resultados finais
# ============================================================================

print("\n" + "="*70)
print("RESUMO FINAL - LSTM BASELINE")
print("="*70)

final_results = {
    'baseline': [results_sleep, results_wesad]
}

with open('results/baseline_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("\nResultados guardados em 'results/baseline_results.json'")
print("\nPróximo passo: Aplicar Differential Privacy (DP) ao modelo LSTM")

# ============================================================================
# OPCIONAL: Visualizar histórico de treino
# ============================================================================

def plot_training_history(history, dataset_name, save_path):
    """Plotar accuracy e loss durante treino"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title(f'{dataset_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid()
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{dataset_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Gráfico salvo: {save_path}")

plot_training_history(history_sleep, 'Sleep-EDF', 'results/training_history_sleep_edf.png')
plot_training_history(history_wesad, 'WESAD', 'results/training_history_wesad.png')