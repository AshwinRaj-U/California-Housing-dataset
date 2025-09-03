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

@app.route('/predict',methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    data = {key: float(value) for key, value in form_data.items()}
    final_input = pd.DataFrame([data])
    final_output=model.predict(final_input)[0]
    return render_template('home.html',prediction_text="The predicted House price is {}".format(final_output))

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error, return a custom response
    return jsonify(error=str(e)), 500


if __name__=='__main__':
    app.run(debug=True)
