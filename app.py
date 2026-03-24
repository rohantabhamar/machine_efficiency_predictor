from flask import Flask, render_template, request
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "artifacts" / "processed" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Update this mapping if your training encoding was different
OPERATION_MODE_MAP = {
    "Idle": 0,
    "Active": 1,
    "Maintenance": 2
}

LABELS = {
    0: "High",
    1: "Low",
    2: "Medium"
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_class = ""
    error_message = None

    if request.method == "POST":
        try:
            timestamp_str = request.form.get("timestamp", "").strip()
            operation_mode_str = request.form.get("operation_mode", "").strip()

            if not timestamp_str:
                raise ValueError("Timestamp is required.")

            if operation_mode_str not in OPERATION_MODE_MAP:
                raise ValueError("Please select a valid operation mode.")

            dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M")

            operation_mode = OPERATION_MODE_MAP[operation_mode_str]
            temperature_c = float(request.form["temperature_c"])
            vibration_hz = float(request.form["vibration_hz"])
            power_consumption_kw = float(request.form["power_consumption_kw"])
            network_latency_ms = float(request.form["network_latency_ms"])
            packet_loss = float(request.form["packet_loss"])
            quality_defect_rate = float(request.form["quality_defect_rate"])
            production_speed = float(request.form["production_speed"])
            maintenance_score = float(request.form["maintenance_score"])
            error_rate = float(request.form["error_rate"])

            input_data = [
                operation_mode,
                temperature_c,
                vibration_hz,
                power_consumption_kw,
                network_latency_ms,
                packet_loss,
                quality_defect_rate,
                production_speed,
                maintenance_score,
                error_rate,
                dt.year,
                dt.month,
                dt.day,
                dt.hour
            ]

            input_array = np.array(input_data).reshape(1, -1)
            scaled_array = scaler.transform(input_array)
            pred = model.predict(scaled_array)[0]

            prediction = LABELS.get(pred, "Unknown")

            if prediction == "High":
                prediction_class = "high"
            elif prediction == "Medium":
                prediction_class = "medium"
            elif prediction == "Low":
                prediction_class = "low"

        except ValueError as e:
            error_message = f"Invalid input: {e}"
        except Exception as e:
            error_message = f"Something went wrong: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        prediction_class=prediction_class,
        error_message=error_message
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)