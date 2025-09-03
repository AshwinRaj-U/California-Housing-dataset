import pickle
from flask import Flask,request,render_template,app,jsonify,url_for
import numpy as np
import pandas as pd

app=Flask(__name__)
# Load the model
model=pickle.load(open('reg_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    input = pd.DataFrame([data])
    #input=np.array(list(data.values())).reshape(1,-1)
    print(input)
    output=model.predict(input)
    print(output)
    return jsonify({'prediction': output.tolist()})


@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error, return a custom response
    return jsonify(error=str(e)), 500


if __name__=='__main__':
    app.run(debug=True)
