from flask import Flask, request, jsonify
from classifier import getPrediction

app = Flask(__name__)
@app.route("/prediction", methods=["POST"])
def prediction():
  image = request.files.get("digit")
  prediction = getPrediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

if __name__ == "__main__":
  app.run(debug=True)