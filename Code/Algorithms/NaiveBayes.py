import pandas as pd
import matplotlib as plt
import numpy as np

from sklearn import linear_model
from sklearn import cross_validation
from scipy.stats import norm

from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import ensemble


# Generating Features


df = pd.read_csv("cric.csv")
X= pd.DataFrame()
X=pd.concat([X,pd.get_dummies(df['Venue'],prefix='Venue')],axis=1)

X=pd.concat([X,pd.get_dummies(df['SecondTeam'],prefix='SecondTeam')],axis=1)


India=pd.DataFrame()
India=pd.concat([India,df['IndianPlayer1']],axis=1)
India=pd.concat([India,df['IndianPlayer2']],axis=1)
India=pd.concat([India,df['IndianPlayer3']],axis=1)
India=pd.concat([India,df['IndianPlayer4']],axis=1)
India=pd.concat([India,df['IndianPlayer5']],axis=1)
India=pd.concat([India,df['IndianPlayer6']],axis=1)
India=pd.concat([India,df['IndianPlayer7']],axis=1)
India=pd.concat([India,df['IndianPlayer8']],axis=1)
India=pd.concat([India,df['IndianPlayer9']],axis=1)
India=pd.concat([India,df['IndianPlayer10']],axis=1)
India=pd.concat([India,df['IndianPlayer11']],axis=1)

India=pd.concat([pd.get_dummies(India[col]) for col in India], axis=1, keys=India.columns)






Opposite=pd.DataFrame()
Opposite=pd.concat([Opposite,df['OppositePlayer1']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer2']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer3']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer4']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer5']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer6']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer7']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer8']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer9']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer10']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer11']],axis=1)

Opposite=pd.concat([pd.get_dummies(Opposite[col]) for col in Opposite], axis=1, keys=Opposite.columns)







#X['Stadium Win Rate']=df['Stadium Win Rate']
X['FirstToBat']=df['FirstToBat']
#X['Home Field Advantage']=df['Home Field Advantage']
#X['WinLose ratio bat or bowl first against team']=df['WinLose ratio bat or bowl first against team']

X['DewPoint']=df['DewPoint']
X['Humidity']=df['Humidity']
X['WindSpeed']=df['WindSpeed']
X['Temp']=df['Temp']


#X=pd.concat([X,India],axis=1)
#X=pd.concat([X,Opposite],axis=1)


y=df['Winner']




from sklearn import cross_validation

np.random.seed(0)


from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

scores = cross_validation.cross_val_score(clf, X, y, cv=3,scoring='mean_squared_error')
print "Accuracy:" ,(np.mean(np.sqrt(abs(scores))))

