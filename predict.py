import sys
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your news headline here\"")
        return

    # Load trained model
    model = joblib.load("fake_news_model.pkl")

    # Join all command-line args into one string (so multi-word headlines work)
    headline = " ".join(sys.argv[1:])

    # Predict
    prediction = model.predict([headline])[0]

    # Interpret result
    if prediction == 0:
        print("❌ FAKE NEWS")
    else:
        print("✅ REAL NEWS")

if __name__ == "__main__":
    main()
