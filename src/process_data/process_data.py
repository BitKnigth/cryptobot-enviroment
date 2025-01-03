
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

def pre_process_data(data, scaler, cuttoff=0.3):
    [
        train_set,
        test_set
    ] = split_data_set(data, cuttoff)

    train_set.append(
         scaler.fit_transform(train_set[1])
    )
    test_set.append(
        scaler.transform(test_set[1])
        )

    return [
        train_set,
        test_set
    ]

def prepare_data(normalized_data, data, time, seq_len):
    # Function to prepare data for LSTM
    X, y = [], []
    for i in range(len(normalized_data) - seq_len):
        X.append(normalized_data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y).reshape(-1, 1), data[:-seq_len], time[:-seq_len]