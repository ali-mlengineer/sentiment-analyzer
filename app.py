from flask import Flask, render_template, request, redirect, url_for
import pickle


app = Flask(__name__)

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        user_text = request.form["text"]

        vec = vectorizer.transform([user_text])

        result = model.predict(vec)[0]

        if result == 1:
            prediction = "Positive"
        else:
            prediction = "Negative"

        return redirect(
            url_for(
                "home",
                prediction=prediction,
                text=user_text
            )
        )

    prediction = request.args.get("prediction")
    user_text = request.args.get("text", "")

    return render_template(
        "index.html",
        prediction=prediction,
        text=user_text
    )
if __name__ == "__main__":
    app.run(debug=True)