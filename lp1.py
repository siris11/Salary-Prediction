import numpy as np
import pickle
from flask import Flask, request,jsonify, render_template
from flask_bootstrap import Bootstrap

#flask
app =Flask(__name__)

model101 = pickle.load(open('model101.pkl','rb'))

@app.route("/")
def Home():
  return render_template('index.html')


@app.route('/predict', methods =["POST"])

def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model101.predict(final_features)
    output =round(prediction[0],2)
    
    return render_template("index.html", prediction_text = "The Predicted Salary is RS.{}".format(output))
    
    
@app.route('/results',methods=["POST"])
def results():
     data =request.get_json(force=True)
     predict = model101.predict([np.array(list(data.values()))])
     output = predict[0]
     return jsonify(output)        

if __name__ == "__main__":
    app.run(debug=True)
