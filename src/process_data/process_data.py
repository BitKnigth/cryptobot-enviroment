
import numpy as np
import ewtpy

def split_data_set(data, cuttoff_proportion):
    dataCutoff = int(len(data) * cuttoff_proportion)
    print("Cutoff: ", dataCutoff)

    train_data = data[:-dataCutoff]
    test_data = data[-dataCutoff:]

    return [train_data, test_data]

def pre_process_data(data, data_scaler, volume_scaler, cuttoff=0.3):

    normalized_data = data_scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    normalized_volume = volume_scaler.fit_transform(data["Volume"].values.reshape(-1, 1))

    # Combina as colunas normalizadas em um único array
    normalized_combined = np.hstack((normalized_data, normalized_volume))

    timeseries = data.index

    [
        train_set,
        test_set
    ] = split_data_set(normalized_combined, timeseries, cuttoff)

    return [
        train_set,
        test_set
    ]

def pre_process_ewt_data(data, data_scaler, volume_scaler, cuttoff=0.3):
    
    normalized_data = data_scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    normalized_volume = volume_scaler.fit_transform(data["Volume"].values.reshape(-1, 1))

    # Combina as colunas normalizadas em um único array
    normalized_combined = np.hstack((normalized_data, normalized_volume))

    timeseries = data.index

    [
        train_data,
        test_data
    ] = split_data_set(data["Close"].values.reshape(-1,1), cuttoff)
    
    [
        train_normal,
        test_normal
    ] = split_data_set(normalized_combined, cuttoff)

    [
        train_timeseries,
        test_series
    ] = split_data_set(timeseries, cuttoff)

    return [
        [train_timeseries, train_data, train_normal],
        [test_series, test_data, test_normal]
    ]

def prepare_data(normalized_data, data, time, seq_len):
    # Function to prepare data for LSTM
    X, y = [], []
    for i in range(len(normalized_data) - seq_len):
        X.append(normalized_data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y).reshape(-1, 1), data[:-seq_len], time[:-seq_len]

def prepare_ewt_data(normalized_data, data, time, seq_len):
    # Function to prepare data for LSTM
    X, y = [], []
    for i in range(len(normalized_data) - seq_len):
        ewt, mfb, boundaries = ewtpy.EWT1D(normalized_data[i:i+seq_len].flatten(), N=13)
        X.append(ewt)
        y.append(data[i+seq_len])
    return np.array(X), np.array(y).reshape(-1, 1), data[:-seq_len], time[:-seq_len]