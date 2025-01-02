
import numpy as np

def split_data_set(data, cuttoff_proportion):
    dataCutoff = int(len(data) * cuttoff_proportion)
    print("Cutoff: ", dataCutoff)

    train_date_series = data.index[:-dataCutoff]
    test_date_series = data.index[-dataCutoff:]

    train_data = data["Close"][:-dataCutoff].values.reshape(-1, 1)
    test_data = data["Close"][-dataCutoff:].values.reshape(-1, 1)

    return [
        [train_date_series, train_data],
        [test_date_series, test_data]
    ]

def normalize_data(data, scaler):
    return scaler.fit_transform(data)

def pre_process_data(data, scaler, cuttoff=0.3):
    [
        train_set,
        test_set
    ] = split_data_set(data, cuttoff)

    test_set.append(
        normalize_data(test_set[1], scaler)
        )
    train_set.append(
        normalize_data(train_set[1], scaler)
    )

    return [
        train_set,
        test_set
    ]

def prepare_data(normalized_data, data, time, seq_len):
    # Function to prepare data for LSTM
    X, y = [], []
    for i in range(seq_len, len(normalized_data)):
        X.append(normalized_data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y).reshape(-1, 1), data[seq_len:], time[seq_len:]