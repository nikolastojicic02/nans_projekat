import pandas as pd
from sklearn.model_selection import train_test_split
from utils_nans1 import *
import matplotlib.pyplot as plt
import seaborn as sb


matplotlib.rcParams['figure.figsize'] = (8, 3)
sb.set(font_scale=1.)

df = pd.read_csv('./BodyFat2.csv')
#print(df)
print(df.isnull().sum())

df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
df = df.drop(columns=['Original'])
#print(df)

x = df.drop(columns=['BodyFat', 'Chest', 'Ankle', 'Weight', 'Hip', 'Thigh', 'Biceps'])
y = df['BodyFat']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)
model = get_fitted_model(x_train, y_train)

corr = df.corr()
plt.figure(figsize=(10, 8))  # Adjust the size as per your requirement
sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

r2 = get_rsquared_adj(model, x_test, y_test)
print(f"\nR2 adjusted LINEARNA REGRESIJA: {r2}\n")
print(model.summary())


# PRETPOSTAVKE
print('\nPRETPOSTAVKE: ')
#print(are_assumptions_satisfied(model, x_train, y_train))
x_with_const = sm.add_constant(x)
print('Za pouzdanost od 95%: ', end='')
is_linearity_found, p_value = linear_assumption(model, x_with_const, y, plot=False)
if is_linearity_found: 
    print('veza je linearna')
else: 
    print('veza je mozda nelinearna, a mozda linearna')


autocorrelation, dw_value = independence_of_errors_assumption(model, x_with_const, y, plot=False)
if autocorrelation is None: 
    print('Nezavisne su greske. Pretpostavka je zadovoljena.')
else: 
    print('Nisu nezavisne greske. Pretpostavka nije zadovoljena.')

n_dist_type, p_value = normality_of_errors_assumption(model, x_with_const, y, plot=False)
if n_dist_type == 'normal':
    print('Greske normalno rasporedjne')
else:
    print('greske nisu normalno rasporedjne')

e_dist_type, p_value = equal_variance_assumption(model, x_with_const, y, plot=False)
if e_dist_type == 'equal':
    print('Konstantna varijansa grešaka. Pretpostavka je zadovoljena.')
else:
    print('Nije konstantna varijansa grešaka. Pretpostavka nije zadovoljena.')

has_perfect_collinearity = perfect_collinearity_assumption(x_with_const, plot=False)
if has_perfect_collinearity == False:
    print('Nema kolinearnosti izmedju nezavisnih promenljivih (90%). Pretpostavka je zadovoljena.')
else:
    print('Ima kolinearnosti izmedju nezavisnih promenljivih (90%). Pretpostavka nije zadovoljena.')

#NOVI MODEL RIDGE

from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)
model_ridge = Ridge()
model_ridge.fit(x_train, y_train)
predictions_ridge = model_ridge.predict(x_test)

print('\nNOVI MODEL RIDGE: ')
r2 = get_rsquared_adj_r(model_ridge, x_test, y_test)
print(f"R2 adjusted RIDGE: {r2} \n")

#SKALIRANJE
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#0.6244531981040997, alpha=3.1
model_ridge = Ridge(alpha=3.1)
model_ridge.fit(x_train_scaled, y_train)
predictions_ridge = model_ridge.predict(x_test_scaled)

r2_adjusted = get_rsquared_adj_r(model_ridge, x_test_scaled, y_test)
print(f"R2 adjusted RIDGE with scaling: {r2_adjusted}")

#RANDOM FOREST
#losiji R2 adj, samo za poredjenje sluzi
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(x_train, y_train)
# Predviđanja modela Random Forest
predictions_rf = model_rf.predict(x_test)
r2 = get_rsquared_adj_r(model_rf, x_test, y_test)
print(f"\nR2 adjusted RANDOM FOREST: {r2}")

