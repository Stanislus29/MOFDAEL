## Question
Give the line currents and line voltages of a three phase transmission line, can we train a model to predict if there is a fault across the lines G, A, B, C, where G is the Ground/earth and A,B,C are the phase lines, at some location given in the dataset, where location is not a parameter to be trained on. 

## Algorithm

We read our csv as 

```python
pd.readcsv(data.csv)
```

The features we are training are:

FEATURE_COLUMNS = [
    "Ia",
    "Ib",
    "Ic",
    "Va",
    "Vb",
    "Vc",
]

TARGET_COLUMNS = [
    "G",
    "C",
    "B",
    "A",
]

$$
\forall
$$