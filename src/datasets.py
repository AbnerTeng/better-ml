from bunch import Bunch
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def load_toy() -> Bunch:
    """
    load the toy dataset
    """
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }

    df = pd.DataFrame(data)
    oe = OrdinalEncoder()
    df_encoded = oe.fit_transform(df)

    toy_data = {
        "data": df_encoded[:, :-1],
        "target": df_encoded[:, -1].astype(int)
    }

    toy_dataset = Bunch()
    toy_dataset.update(toy_data)

    return toy_dataset