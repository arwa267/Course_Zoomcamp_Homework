
FROM python:3.8.12-slim


RUN pip install pipenv 

WORKDIR /predict


COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY["predict.py","Random_forest_model_depth=13_and_number_leaf=80.bin","./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve","--listen=0.0.0.0:9696","predict:app"]








