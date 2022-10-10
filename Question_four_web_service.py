

import pickle
from flask import Flask
from flask import request
from flask import jsonify




file_model="model1.bin"
file_dv="dv.bin"




with open(file_model,'rb') as f_in:
    model=pickle.load(f_in)




with open(file_dv,'rb') as f_in:
    dv=pickle.load(f_in)

    
app=Flask('predict')
@app.route('/predict',methods=['POST'])

def predict():
    client=request.get_json()
    X=dv.transform([client])
    prob=model.predict_proba(X)[0,1]
    decision=prob>=0.5
    result={'probability':float(prob),'decision':bool(decision)}
    return jsonify(result)
    

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0',port=96969)









