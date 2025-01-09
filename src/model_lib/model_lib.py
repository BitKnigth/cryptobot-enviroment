from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout

def model_builder(units, dropout_rate, dense_units, shape):
    model = Sequential([
        Input(shape),
        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units),
        Dense(dense_units),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    return model

def optimize_hyperparameters(model_builder, X_train, y_train, X_val, y_val, optuna, n_trials=50):
    """
    Otimiza os hiperparâmetros de um modelo usando Optuna.

    Parâmetros:
    - model_builder: função que constrói e retorna um modelo Keras compilado.
                     Essa função deve aceitar os hiperparâmetros como argumento.
    - X_train, y_train: Dados de treinamento.
    - X_val, y_val: Dados de validação.
    - n_trials: Número de tentativas de otimização (padrão = 50).

    Retorna:
    - Melhor conjunto de hiperparâmetros encontrados pelo Optuna.
    """
    def objective(trial):
        # Sugerir hiperparâmetros a serem otimizados
        units = trial.suggest_int("units", 32, 128, 256)  # Neurônios por camada
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3, 0.5)  # Taxa de Dropout
        dense_units = trial.suggest_int("dense_units", 8, 16, 64)  # Neurônios na camada densa
        
        # Construir o modelo usando a função fornecida
        model = model_builder(units, dropout_rate, dense_units)
        
        # Treinar o modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            epochs=trial.suggest_int("epochs", 10, 50, 100, 150),
            verbose=1
        )
        
        # Retornar a métrica de validação (por exemplo, perda)
        val_loss = history.history["val_loss"][-1]
        return val_loss

    # Criar o estudo do Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Retornar os melhores hiperparâmetros encontrados
    return study.best_params
