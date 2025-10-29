
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import streamlit as st
from PIL import Image

dataset = pd.read_csv("card_transdata.csv")


dataset.isnull().sum()
dataset.duplicated()
dataset = dataset.drop_duplicates()

dataset.describe()
dataset.info()



x = dataset.iloc[:,:-1]
y = dataset["fraud"]

y.value_counts()



ru = RandomUnderSampler()
ru_x,ru_y = ru.fit_resample(x,y)

ru_y.value_counts()



# lo = LogisticRegression()
# lst = []
# for i in range(1,100):
#     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=i)
#     lo.fit(x_train,y_train)
#     train_score = lo.score(x_train,y_train)*100
#     test_score = lo.score(x_test,y_test)*100
#     if train_score >= test_score >=train_score-2:
#         lst.append([i,"  ",train_score,"  ",test_score])

# for i in lst:
#     print(i)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

lo = LogisticRegression(class_weight={1:10})
lo.fit(x_train,y_train)

train_score = lo.score(x_train,y_train)*100
test_score = lo.score(x_test,y_test)*100


x_train_pred = lo.predict(x_train)
x_test_pred = lo.predict(x_test)

train_acc = accuracy_score(y_train,x_train_pred)*100
test_acc = accuracy_score(y_test,x_test_pred)*100

train_cm = confusion_matrix(y_train,x_train_pred)
sns.heatmap(train_cm,annot=True)
plt.show()

test_cm = confusion_matrix(y_test,x_test_pred)
sns.heatmap(test_cm,annot=True)
plt.show()





data = (5.609998089130593,0.1375131862187362,0.7612338492848543,1.0,0.0,0.0,1.0)
np_data = np.asarray(data)
reshape_data = np_data.reshape(1,-1)

ans = lo.predict(reshape_data)
if ans[0] == 1:
    print(ans[0],": This is fraudulent Transaction")
else:
    print(ans[0],": This is legitimate Transaction")

def app():
    st.title("Credit Card Fraud")
    dist_from_home = st.number_input("Enter distance from home to transaction location in kilometer")
    dist_last_2_trans = st.number_input("Enter distance between last 2 transaction in kilometer")
    ratio = st.number_input("Enter ratio between last all transaction and this one ")
    repeat_retailer = st.number_input("Purchased before from the same retailer/store (1 = yes, 0 = no)",max_value=1,min_value=0)
    used_chip = st.number_input("Using a physical card chip (1 = yes, 0 = no)",max_value=1,min_value=0)
    used_pin = st.number_input("PIN was entered for the transaction (1 = yes, 0 = no)",max_value=1,min_value=0)
    online_order = st.number_input("Transaction was made online (1 = yes, 0 = no)",max_value=1,min_value=0)


    if st.button("Predict"):
        data = (dist_from_home, dist_last_2_trans, ratio, repeat_retailer, used_chip, used_pin, online_order)
        np_data = np.asarray(data)
        reshape_data = np_data.reshape(1, -1)

        ans = lo.predict(reshape_data)
        if ans[0] == 1:
            st.warning(f"{ans[0]} : This is fraudulent Transaction")
        else:
            st.success(f"{ans[0]} : This is legitimate Transaction")

if __name__ == "__main__":
    app()