import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image
import pydotplus

col_names = ['ID','Class','Gender','Age','Siblings','Parch','Survived']

df = pd.read_csv('data.csv',names=col_names).iloc[1:]
print(df.head())

features = ['ID','Class','Gender','Age','Siblings','Parch','Survived']
X = df[features]
Y = df.Survived

X_train , X_test , Y_train, Y_test =  train_test_split(X,Y,test_size=0.3,random_state=1)

tree = DecisionTreeClassifier()

tree = tree.fit(X_train,Y_train)

y_pred = tree.predict(X_test)

dot_data = StringIO() 

export_graphviz(tree, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['0','1'])

print(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Survived.png')
Image(graph.create_png())