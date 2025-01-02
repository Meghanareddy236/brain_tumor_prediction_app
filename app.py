from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
try:
    stroke_model = joblib.load("model.joblib")
except FileNotFoundError:
    raise Exception("Model file 'model.joblib' not found. Ensure it exists in the project directory.")

# Helper function for prediction
def predict_input(single_input):
    try:
        input_df = pd.DataFrame([single_input])
        encoded_cols, numeric_cols = stroke_model["encoded_cols"], stroke_model["numeric_cols"]
        preprocessor = stroke_model["preprocessor"]
        input_df[encoded_cols] = preprocessor.transform(input_df)
        X = input_df[numeric_cols + encoded_cols]
        prediction = stroke_model['model'].predict(X)
        return prediction
    except Exception as e:
        raise ValueError(f"Error in prediction process: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get form data
            gender = request.form["gender"].lower()
            age = int(request.form["age"])
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            ever_married = request.form["ever_married"].lower()
            work_type = request.form["work_type"]
            residence_type = request.form["residence_type"]
            avg_glucose_level = float(request.form["avg_glucose_level"])
            bmi = float(request.form["bmi"])
            smoking_status = request.form["smoking_status"].lower()

            # Validate inputs
            if age < 0 or avg_glucose_level < 0 or bmi < 0:
                return render_template("index.html", error="Age, glucose level, and BMI must be non-negative values.")

            # Map work type
            work_type_mapping = {
                "Government job": "Govt_job",
                "Children": "children",
                "Never Worked": "Never_worked",
                "Private": "Private",
            }

            single_input = {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type_mapping.get(work_type, work_type),
                "Residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status,
            }

            # Predict and redirect to result page
            prediction = predict_input(single_input)
            result = "Likely" if prediction[0] == 1 else "Not Likely"
            return render_template("result.html", result=result)

        except Exception as e:
            return render_template("index.html", error=f"Error processing request: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
