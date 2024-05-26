# !pip install flask-ngrok

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/refactor', methods=['GET'])
def refactor_code():
    data = request.json
    code = data['code']
    # Your refactoring logic here
    refactored_code = "refactored code"  # Replace with actual refactoring logic
    return jsonify({'refactored_code': refactored_code})

app.run()
