
from flask import Flask
from flask import request
from flask import jsonify
import pickle
import os



pickle_file='Random_forest_model_depth=13_and_number_leaf=80.bin'

with open(pickle_file,'rb') as f_in:
    dv,model=pickle.load(f_in)
    
app=Flask('Potability')

@app.route('/predict',methods=['POST'])
def predict():
    inp_user=request.get_json()
    x=dv.transform([inp_user])
    y_pred=model.predict(x)
    print(y_pred)
    result={'result':bool(y_pred)}
    return jsonify(result)
    
    
    
  

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)

    



    