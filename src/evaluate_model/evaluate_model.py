import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from src import prepare_data
def prepare_data_and_predict(model, prepare_data_function, normalized_test_data, test_data, test_date_series, seq_len):
    """
    Prepara os dados de teste e realiza previsões no conjunto de teste.
    :param model: Modelo treinado que será utilizado para realizar as previsões.
    :param normalized_test_data: Dados normalizados para entrada no modelo.
    :param test_data: Dados de teste originais para comparação.
    :param test_date_series: Série temporal das datas correspondentes aos dados de teste.
    :param seq_len: Comprimento da sequência utilizada no modelo.
    :return: Dados de entrada de teste, valores reais, previsões e série temporal de datas.
    """
    # Prepara os dados para o modelo com base no comprimento da sequência
    X_test, y_test, test_data, test_date_series = prepare_data_function(
        normalized_test_data, test_data, test_date_series, seq_len
    )

    # Realiza as previsões usando o modelo treinado
    predictions = model.predict(X_test)

    # Ajusta a forma das previsões para uma única coluna
    reshaped_predictions = predictions.reshape(-1, 1)

    return X_test, y_test, reshaped_predictions, test_date_series

def plot_predictions(test_date_series, y_test, test_predictions, title="Prediction vs Actual"):
    """
    Gera um gráfico das previsões e valores reais.
    :param test_date_series: Série temporal das datas correspondentes aos dados de teste.
    :param y_test: Valores reais dos dados de teste.
    :param test_predictions: Valores previstos pelo modelo.
    :param title: Título do gráfico.
    """
    # Configura o tamanho do gráfico
    plt.figure(figsize=(12, 6))

    # Plota os valores reais
    plt.plot(test_date_series[:], y_test[:], label='Actual Price')

    # Plota os valores previstos
    plt.plot(test_date_series[:], test_predictions[:], label='Predictions on Test set')

    # Configurações do gráfico
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()

    # Exibe o gráfico
    plt.show()

def evaluate_and_save_metrics(y_test, test_predictions, model_name, output_folder="evaluation_metrics"):
    """
    Avalia as métricas do modelo, exibe os resultados, e salva as métricas em um arquivo JSON.
    :param y_test: Valores reais dos dados de teste.
    :param test_predictions: Valores previstos pelo modelo.
    :param model_name: Nome do modelo para identificar o arquivo de saída.
    :param output_folder: Diretório onde as métricas serão salvas.
    :return: Dicionário contendo as métricas calculadas.
    """
    # Calcula o erro médio quadrático
    mse = mean_squared_error(y_test, test_predictions)

    # Calcula a raiz do erro médio quadrático
    rmse = np.sqrt(mse)

    # Calcula o erro absoluto médio
    mae = mean_absolute_error(y_test, test_predictions)

    # Calcula o erro percentual absoluto médio
    mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100

    # Calcula o coeficiente de determinação (R²)
    r2 = r2_score(y_test, test_predictions)

    # Armazena as métricas em um dicionário
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }

    # Exibe as métricas calculadas
    print(f"Model: {model_name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Verifica se a pasta de saída existe, cria caso contrário
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define o caminho do arquivo JSON para salvar as métricas
    file_path = os.path.join(output_folder, f"evaluation_{model_name}.json")

    # Salva as métricas no arquivo JSON
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics

def full_evaluation_flow(model, prepare_data_function, normalized_test_data, test_data, test_date_series, seq_len, model_name, output_folder="evaluation_metrics"):
    """
    Abstrai o fluxo completo de prever, plotar e avaliar o modelo.
    :param model: Modelo treinado que será utilizado para realizar as previsões.
    :param normalized_test_data: Dados normalizados para entrada no modelo.
    :param test_data: Dados de teste originais para comparação.
    :param test_date_series: Série temporal das datas correspondentes aos dados de teste.
    :param seq_len: Comprimento da sequência utilizada no modelo.
    :param model_name: Nome do modelo para identificar o arquivo de saída.
    :param output_folder: Diretório onde as métricas serão salvas.
    :return: Dicionário contendo as métricas calculadas.
    """
    # Etapa 1: Prepara os dados e realiza as previsões
    _, y_test, test_predictions, test_date_series = prepare_data_and_predict(
        model, prepare_data_function, normalized_test_data, test_data, test_date_series, seq_len
    )

    # Etapa 2: Gera o gráfico de previsões
    plot_predictions(test_date_series, y_test, test_predictions, title=f"{model_name} Predictions")

    # Etapa 3: Avalia e salva as métricas
    metrics = evaluate_and_save_metrics(y_test, test_predictions, model_name, output_folder)

    return metrics

# Uso exemplo (Assumindo que as funções `prepare_data` e `model.predict` existam):
# metrics = full_evaluation_flow(model, normalized_test_data, test_data, test_date_series, seq_len, "Ripple_Model")
