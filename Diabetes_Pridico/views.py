from django.shortcuts import render

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    try:
        data = pd.read_csv(r"E:\OneDrive - Birla Institute of Technology\Desktop\programs code\diabetes pridiction system\diabetes.csv")

        x = data.drop("Outcome", axis=1)
        y = data['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

        model = LogisticRegression()
        model.fit(x_train, y_train)

        val1 = float(request.GET.get('n1', 0.0))
        val2 = float(request.GET.get('n2', 0.0))
        val3 = float(request.GET.get('n3', 0.0))
        val4 = float(request.GET.get('n4', 0.0))
        val5 = float(request.GET.get('n5', 0.0))
        val6 = float(request.GET.get('n6', 0.0))
        val7 = float(request.GET.get('n7', 0.0))
        val8 = float(request.GET.get('n8', 0.0))

        pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

        result1 = "Positive" if pred == 1 else "Negative"
    except ValueError:
        result1 = "Invalid input. Please enter numeric values."

    return render(request, 'predict.html', {"result2": result1})

def liver_predict(request):
    return render(request, 'liver_predict.html')