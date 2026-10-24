import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

from src.Preprocessing import PrepareData, StandardizeFeatures, TrainTestSplit
from src.Linear_Regression import LinearRegression
from src.polynomial_regression import PolynomialRegressionScratch


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    print("=== Battery Temperature Predictor (Linear vs Polynomial) ===")

    data_path = "data/battery_data.csv"
    df = pd.read_csv(data_path, sep=";", encoding="latin1")

    X, y = PrepareData(df)
    X_train, X_test, y_train, y_test = TrainTestSplit(X, y, test_ratio=0.2, seed=42)

    X_train_norm, mean, std = StandardizeFeatures(X_train)
    X_test_norm = (X_test - mean) / std

    lin_model = LinearRegression(learning_rate=0.05, iterations=2000)
    lin_model.fit(X_train_norm, y_train)
    y_pred_lin = lin_model.predict(X_test_norm)
    rmse_lin = rmse(y_test, y_pred_lin)

    poly_model = PolynomialRegressionScratch(degree=2, learning_rate=0.05, iterations=2000)
    poly_model.fit(X_train_norm, y_train)
    y_pred_poly = poly_model.predict(X_test_norm)
    rmse_poly = rmse(y_test, y_pred_poly)

    fig = plt.figure(figsize=(13.5, 7.5))
    plt.subplots_adjust(bottom=0.32, wspace=0.25)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(
        range(len(lin_model.cost_history)),
        lin_model.cost_history,
        color="tab:blue",
        lw=2,
        label=f"Linear Cost"
    )
    ax1.plot(
        range(len(poly_model.cost_history)),
        poly_model.cost_history,
        color="tab:red",
        lw=2,
        linestyle="--",
        label=f"Polynomial Cost"
    )
    ax1.set_title("Gradient Descent Convergence Comparison", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost (MSE / 2)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    y_test_flat = y_test.reshape(-1)
    ax2.scatter(
        y_test_flat, y_pred_lin.reshape(-1),
        alpha=0.3, color="tab:orange", s=15,
        label=f"Linear (RMSE: {rmse_lin:.3f})"
    )
    ax2.scatter(
        y_test_flat, y_pred_poly.reshape(-1),
        alpha=0.3, color="tab:purple", s=15,
        label=f"Polynomial (RMSE: {rmse_poly:.3f})"
    )

    min_val = float(min(np.min(y_test), np.min(y_pred_lin), np.min(y_pred_poly)))
    max_val = float(max(np.max(y_test), np.max(y_pred_lin), np.max(y_pred_poly)))
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Ideal Fit")

    ax2.set_title("Actual vs Predicted Comparison", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Actual Temp [°C]")
    ax2.set_ylabel("Predicted Temp [°C]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    default_v = f"{mean[0]:.1f}"
    default_i = f"{mean[1]:.1f}"
    default_a = f"{mean[2]:.1f}"
    default_t = f"{mean[3]:.1f}"

    ax_v = plt.axes([0.08, 0.18, 0.14, 0.06])
    tb_v = TextBox(ax_v, "Voltage [V]: ", initial=default_v)

    ax_i = plt.axes([0.30, 0.18, 0.14, 0.06])
    tb_i = TextBox(ax_i, "Current [A]: ", initial=default_i)

    ax_a = plt.axes([0.52, 0.18, 0.14, 0.06])
    tb_a = TextBox(ax_a, "Ambient [°C]: ", initial=default_a)

    ax_t = plt.axes([0.74, 0.18, 0.14, 0.06])
    tb_t = TextBox(ax_t, "Time [s]: ", initial=default_t)

    ax_btn = plt.axes([0.40, 0.06, 0.20, 0.07])
    btn = Button(ax_btn, "Predict Both", color="lightgreen", hovercolor="0.85")

    result_text = fig.text(
        0.75, 0.09,
        "Predictions -- Linear: -- °C | Poly: -- °C",
        fontsize=9,
        fontweight="bold",
        color="darkblue"
    )

    def on_predict(event):
        try:
            v_val = float(tb_v.text)
            i_val = float(tb_i.text)
            a_val = float(tb_a.text)
            t_val = float(tb_t.text)

            raw_in = np.array([[v_val, i_val, a_val, t_val]], dtype=np.float64)
            norm_in = (raw_in - mean) / std

            pred_lin = float(lin_model.predict(norm_in)[0, 0])
            pred_poly = float(poly_model.predict(norm_in)[0, 0])

            result_text.set_text(f"Predictions -- Linear: {pred_lin:.2f} °C | Poly: {pred_poly:.2f} °C")
            fig.canvas.draw_idle()
        except ValueError:
            result_text.set_text("Error: Invalid numbers!")
            fig.canvas.draw_idle()

    btn.on_clicked(on_predict)

    on_predict(None)

    plt.show()


if __name__ == "__main__":
    main()
