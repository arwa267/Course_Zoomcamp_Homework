# -*- coding: utf-8 -*-

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

 
model_rf=bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

model_runner=model_rf.to_runner()

svc=bentoml.Service("model_zoomcamp",runners=[model_runner])

@svc.api(input=NumpyNdarray(),output=JSON())
def classify(vector):
    prediction=model_runner.predict.run(vector)
    print(prediction)
