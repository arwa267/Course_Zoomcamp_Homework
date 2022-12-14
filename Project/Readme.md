
                                                  ---------------------------------- Potability of Water---------------------

Let's say you are very thirsty and you have passed through a water spring . Unfortunately, you should make sure whether you can drink from this water 
since it may be  poisonous.
Therefore,Let use the power of Machine Learning to know whether the water is potable or not.

For this case, I have used the  Potability data from Dphi Community website that shares with all Data Science Lovers free datasets  to  practice on. 
Link to the dataset(https://dphi.tech/challenges/253/overview/about) ( also attached in the github folder)


Dataset Description:

The dataset is made up of 10 Columns 
9 Numerical Columns that quantifies the different Characteristics of the Water  Spring that are :

-PH
-Hardness
-Solids 
-Chloramin
-Sulfate 
-Organic Carbon
-Trihalomethanes

The last column named "Potability" is a boolean column that is 1 when the water is Potable and 0 when it is not.
 


Therefore my problem is a Classification Problem where I am going to use the values of the  9 different Numerical Columns
to predict wheter the water is potable or not.


I trained both Random Forests and logistic regression on the Training Data and I tuned their Parameters ( Number of Trees , max depth for Random Forest ,
C for logistic Regeression).
At the end, I choose Random forests to be my predicting model (I deployed this model) since it performed better on the validation date.

-----------------------------------------------------------------------Contents of the Projects----------------------------------------------------

1) The Training Procedure is present in notebook.ipynb  where I loaded  the Data , Performed EDA , splitted the data, 
 and Trained both Models along with Hyper Parameters

2) The Random Forest Model (with the optimal Parameters) is trained and saved in a pickle file  
entitled as "Random_forest_model_depth=13_and_number_leaf=80.bin" in the train.py file

3) The model is deployed using Flask in the python script entitled as "predict.py"

4)if you want to the run the service on your computer download "Random_forest_model_depth=13_and_number_leaf=80.bin" 
and  "predict.py" ,put them in the same folder then run python predict.py

5)Since Flask is a  development server, I will also deploy the model using  Waitress which is a production server, 
and I have recorded a video to show you how I deployed the model using both Flask
and Waitress and how I interacted with the server via the requests library via jupyter notebook .

You can access the video through (https://drive.google.com/file/d/1UjOSi96iBtlq26ZvYQubRZTpejHgsxS7/view?usp=sharing)
and the notebook is attached her with name "Connecting _with _The_ server.ipynb"


6)I have created the a virtual enviroment for the project using pipenv and all the requirments are stated in the Pipfile, 
so if you want to run all the dependencies just make sure that you have installed the pipenv package , download the pipfile into the directory that you want
and just type:
                                              pipenv install
you will notice after the installation is done that a Pipfile.lock (same as the present) is created as well

7)Docker enables us to  isolate our model entirely from all the other application. 
I wrote  a Docker file for the model and i have started from the refrence image "python:3.8.12-slim". Also, I have copied all the required file such as 
"Pipfile" ,"Pipfile.lock" ,""Random_forest_model_depth=13_and_number_leaf=80.bin", "predict.py" that are required for the deployment.

8) in order to run the service using the docker file then place the Dockerfile to the directory that you have placed all other files, then type 
 
     docker build -t Project .

     where you can use any name other than Project
 
    then to run the service run :
    
    docker run -it --rm -p 9696:9696 Project

9)now if you run the jupyter notbook "Connecting _with _The_ server.ipynb", you will get the same results as that of the video.
    

