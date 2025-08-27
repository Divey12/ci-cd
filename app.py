#app.py
import os
import json
import joblib
import numpy as np
from flask import flask, request, jsonify

# ---Config ---
MODEL_PATH = os.getenv("MODEL_PATH","model/iris_model.pikl") #adjust filename if needed

# --- App ---
app = flask(__name__)

#Load once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    #Fail fast with  a helpful error message
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")
@app.get('/health')

def health():
    return jsonify(status='ok'), 200

@app.post('/predict')
def predict():
    """Axxepts either:
    {"input":[[...feature vectors...]]} or
    {"input":[{"feature1":value1,"feature2":value2,...},...]}
    """	
    try:
	payload =  request.get_json(force=True)
	x = payload.get("input")	
	if x is None:
		return jsonify(error = "Missing 'input'"), 400
	
	#Normalize to 2D array
	if isinstance(x,list) and (len(x)>0) and not isinstance(x[0],list):
	    x = [x]
	x= np.array(x, dtype = float)
	preds = model.predict(x)
	# If your model returns numpy types, convert to python
	preds = preds.tolist()
	return jsonify(predictions = preds),200
    except Exception as e:
	return jsonify (error = str(e)),500
if__name__ == "__main__:
    #Local dev only;Render will run with Gunicorn (see start(Command below))
    app.run(host = "0.0.0.0", port = int(os.environ.get("PORT",8000)))
   
	