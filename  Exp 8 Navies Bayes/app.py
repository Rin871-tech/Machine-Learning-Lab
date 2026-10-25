from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)


# ============================================================
# DATASET
# ============================================================

data = [
    ("Buy one get one free. Shop now and save big.", "Promotional"),
    ("The attendance report will be finalized this week.", "Important"),
    ("Reminder: the coding contest closes tonight, register before 11 PM.", "Important"),
    ("The classroom for today's lecture has changed to Room 204.", "Important"),
    ("The placement aptitude test is scheduled for Saturday.", "Important"),
    ("Students selected for the internship should report to the training office.", "Important"),
    ("The student portal is under maintenance from 8 PM to 10 PM.", "Important"),
    ("Please confirm your participation in the workshop by tomorrow.", "Important"),
    ("Mega sale starts now. Up to 70 percent discount.", "Promotional"),
    ("The coding contest registration closes tonight.", "Important"),
    ("The seminar on Generative AI starts at 2 PM in Seminar Hall.", "Important"),
    ("Special student discount available. Click to activate.", "Promotional"),
    ("Free recharge available for selected users. Claim now.", "Promotional"),
    ("Special offer just for you. Click here before it ends.", "Promotional"),
    ("You are selected for a special cashback reward.", "Promotional"),
    ("Your exclusive bonus is ready. Click now to claim it.", "Promotional"),
    ("Your internal assessment marks are available on the portal.", "Important"),
    ("The timetable for the next semester is now available.", "Important"),
    ("Your lucky number has won a shopping voucher.", "Promotional"),
    ("Bring your laptop for tomorrow's Python practical.", "Important"),
    ("Exclusive offer! Get a free gift voucher today.", "Promotional"),
    ("Students must submit the project synopsis by Wednesday.", "Important"),
    ("Unlock your exclusive coupon and save money today.", "Promotional"),
    ("Your project presentation slot is at 10:30 AM.", "Important"),
    ("You have been chosen for a surprise reward.", "Promotional"),
    ("The department meeting for student representatives is on Friday.", "Important"),
    ("Tomorrow's Machine Learning lecture is shifted to 11 AM.", "Important"),
    ("Congratulations! You won a free smartphone. Claim it now.", "Promotional"),
    ("Congratulations! You qualify for a bonus reward.", "Promotional"),
    ("The examination form must be submitted before the deadline.", "Important"),
    ("Hurry! Your chance to win a brand new laptop ends tonight.", "Promotional"),
    ("Get instant cashback on your next purchase.", "Promotional"),
    ("Get 80 percent off on shoes. Limited time offer!", "Promotional"),
    ("Your viva will be conducted after the practical examination.", "Important"),
    ("Flash sale! Grab your favorite products before midnight.", "Promotional"),
    ("Claim your free movie tickets before they expire.", "Promotional"),
    ("Act now to receive your complimentary gift.", "Promotional"),
    ("The deadline for course registration is tomorrow.", "Important"),
    ("Get premium membership at an unbelievable price.", "Promotional"),
    ("A special discount is waiting for you. Shop before midnight.", "Promotional"),
    ("The NLP practical will be conducted in Lab 3 tomorrow.", "Important"),
    ("Win exciting prizes by clicking this lucky link.", "Promotional"),
    ("Limited offer! Free delivery on all orders today.", "Promotional"),
    ("The university exam timetable has been published.", "Important"),
    ("Congratulations winner! Send your details to receive the prize.", "Promotional"),
    ("Win a travel package by participating in our contest.", "Promotional"),
    ("Attendance shortage students should meet the class coordinator.", "Important"),
    ("Congratulations on your achievement! Collect your certificate from the office.", "Important"),
    ("Download the question bank from the LMS before the next lecture.", "Important"),
    ("Please check your university email for the internship instructions.", "Important"),
    ("The faculty has uploaded notes for the next unit.", "Important"),
    ("Bring your identity card for the campus recruitment drive.", "Important"),
    ("Don't miss our biggest sale of the year.", "Promotional"),
    ("Submit the Deep Learning assignment before Friday 5 PM.", "Important"),
    ("The project review is scheduled for Monday morning.", "Important"),
    ("Earn easy money from home. Join our program today.", "Promotional"),
]


# Convert dataset into DataFrame
df = pd.DataFrame(data, columns=["message", "label"])


# ============================================================
# TRAIN MODEL
# ============================================================

X = df["message"]
y = df["label"]


# Same Bag-of-Words configuration as notebook
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)


# Convert text into numerical features
X_bow = vectorizer.fit_transform(X)


# Create Naive Bayes classifier
model = MultinomialNB()


# Train model
model.fit(X_bow, y)


print("Campus Message Detective model trained successfully!")
print("Dataset size:", len(df))


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probabilities = None
    message = ""

    if request.method == "POST":

        message = request.form.get("message", "").strip()

        if message:

            # Convert message into Bag-of-Words
            message_vector = vectorizer.transform([message])

            # Make prediction
            prediction = model.predict(message_vector)[0]

            # Get probabilities
            probability_values = model.predict_proba(message_vector)[0]

            # Get class names
            classes = model.classes_

            probabilities = {}

            for class_name, probability in zip(
                classes,
                probability_values
            ):
                probabilities[class_name] = round(
                    probability * 100,
                    2
                )

            # Model confidence
            confidence = max(probability_values) * 100

            result = {
                "prediction": prediction,
                "confidence": round(confidence, 2)
            }

    return render_template(
        "index.html",
        result=result,
        probabilities=probabilities,
        message=message
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )