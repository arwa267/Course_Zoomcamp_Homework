

print("Loading Libraries")
import pandas as pd
import numpy as np
import sklearn
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


print("Loading the Data and Checking for Null Libraries")



df=pd.read_csv("potability_of_water.csv")





print("Dividing the Data For Training, Validating and Testing")
 



df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=2)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=2)




y_train = df_train['Potability']
y_val =df_val['Potability']
y_test =df_test['Potability']




del df_train['Potability']
del df_val['Potability']
del df_test['Potability']





print("Using a Dvictorizer for both Training and Validation Data")



train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)


print("Training the random forest model")


number_of_trees=np.arange(10, 201, 10)
max_depth=np.arange(5,15,2)





scores = np.zeros((len(number_of_trees),len(max_depth)))

for i in range(len(max_depth)):
    for j  in range(len(number_of_trees)):
        rf = RandomForestClassifier(n_estimators=number_of_trees[j],max_depth=max_depth[i],random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        scores[j,i]= roc_auc_score(y_val, y_pred)
        print("for trees=",number_of_trees[j], " depth =", max_depth[i],"score=", scores[j,i])

        




print("Get the parameter with the highest score")
indices=np.where(scores==np.max(scores))




print("The best number of trees is", number_of_trees[indices[0][0]])
print("The best max depth is", max_depth[indices[1][0]])


print("the final model")


optimal_trees=number_of_trees[indices[0][0]]
max_depth_optimal= max_depth[indices[1][0]]
rf = RandomForestClassifier(n_estimators=optimal_trees,max_depth=max_depth_optimal,random_state=1)
rf.fit(X_train, y_train)

# Saving the model in a pickle file
output=f'Random_forest_model_depth={max_depth_optimal}_and_number_leaf={optimal_trees}.bin'

with open(output,'wb') as f:
    pickle.dump((dv,rf),f)
     

