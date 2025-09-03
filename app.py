from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("engagement_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Collect inputs from form
        from_home = int(request.form["from_home"])
        from_hashtags = int(request.form["from_hashtags"])
        from_explore = int(request.form["from_explore"])
        profile_visits = int(request.form["profile_visits"])
        follows = int(request.form["follows"])
        caption = request.form["caption"]
        hashtags = request.form["hashtags"]

        # Feature engineering
        caption_len = len(caption.split())
        hashtag_count = len(hashtags.split())

        # Prediction
        X_new = [[from_home, from_hashtags, from_explore, 0,
                  profile_visits, follows, caption_len, hashtag_count]]
        prediction = int(model.predict(X_new)[0])

        return render_template("index.html", result=f"Predicted Engagement: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)

