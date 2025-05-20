import time
import optuna
import subprocess
import os
import matplotlib.pyplot as plt
import statistics

PARAMS_DIR = ""
LSTM_SCRIPT_PATH = ""


def write_params_to_file(params, filepath):
    with open(filepath, "w") as f:
        f.write("possible_indicators=SMA EMA RSI MACD BB_UP BB_DOWN ATR High Low Open Volume\n")
        f.write("indicators=SMA EMA RSI MACD BB_UP BB_DOWN ATR High Low Open Volume\n")
        f.write(f"window_size={params['window_size']}\n")
        f.write(f"epochs={params['epochs']}\n")
        f.write(f"batch_size={params['batch_size']}\n")
        f.write(f"layers={params['layers']}\n")
        f.write(f"units={params['units']}\n")
        f.write(f"dropout={params['dropout']}\n")


def moving_median(data, window_size):
    medians = []
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        medians.append(statistics.median(window))

    return medians


def objective(trial, fitness_values):
    params = {
        "window_size": trial.suggest_int("window_size", 10, 90),
        "epochs": trial.suggest_int("epochs", 30, 75),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "layers": trial.suggest_int("layers", 1, 5),
        "units": trial.suggest_int("units", 40, 256),
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
    }

    params_path = PARAMS_DIR + 'params.txt'

    write_params_to_file(params, params_path)

    result = subprocess.run(
        ["python3", LSTM_SCRIPT_PATH, params_path],
        cwd=os.path.dirname(LSTM_SCRIPT_PATH),
        capture_output=True,
        text=True,
    )

    fitness_str = result.stdout.strip()
    fitness = float(fitness_str)

    fitness_values.append(fitness)

    return fitness


def plot_fitness(fitness_values):
    median_trend = moving_median(fitness_values, window_size=30)  # Зроблено плавніше

    plt.figure(figsize=(10, 6))
    plt.plot(fitness_values, marker='o', label='Fitness values')
    plt.plot(median_trend, color='orange', linewidth=2, label='Median trend (window=30)')
    plt.xlabel('Measurement Index')
    plt.ylabel('Fitness (%)')
    plt.title('Fitness Values with Smooth Median Trend')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.makedirs(PARAMS_DIR, exist_ok=True)

    start_time = time.time()

    fitness_values = []

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, fitness_values), n_trials=150)

    end_time = time.time()

    print("Best parameters:", study.best_params)
    print("Best error:", study.best_value)
    print("Time taken:", end_time - start_time)

    plot_fitness(fitness_values)