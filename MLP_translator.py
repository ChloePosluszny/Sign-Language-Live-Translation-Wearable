import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, \
mean_squared_error, log_loss, root_mean_squared_error,log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier,MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

dfa = pd.read_csv('training_data/a.csv')
dfb = pd.read_csv("training_data/b.csv")
dfc = pd.read_csv("training_data/c.csv")
dfd = pd.read_csv("training_data/d.csv")
dfe = pd.read_csv("training_data/e.csv")
dflove = pd.read_csv("training_data/i love you.csv")
# dff = pd.read_csv("f.csv")
# dfg = pd.read_csv("g.csv")
# dfh = pd.read_csv("h.csv")
# dfi = pd.read_csv("i.csv")
# dfj = pd.read_csv("j.csv")
# dfk = pd.read_csv("k.csv")
# dfl = pd.read_csv("l.csv")
# dfm = pd.read_csv("m.csv")
# dfn = pd.read_csv("n.csv")
# dfo = pd.read_csv("o.csv")
# dfp = pd.read_csv("p.csv")
# dfq = pd.read_csv("q.csv")
# dfr = pd.read_csv("r.csv")
# dfs = pd.read_csv("s.csv")
# dft = pd.read_csv("t.csv")
# dfu = pd.read_csv("u.csv")
# dfv = pd.read_csv("v.csv")
# dfw = pd.read_csv("w.csv")
# dfx = pd.read_csv("x.csv")
# dfy = pd.read_csv("y.csv")
# dfz = pd.read_csv("z.csv")

# data = pd.concat([dfa, dfb, dfc, dfd,
#                   dfe, dff, dfg, dfh, dfi, dfj, dfk, dfl, dfm, dfn, dfo,
#                     dfp, dfq, dfr, dfs, dft, dfu, dfv, dfw, dfx, dfy, dfz,], ignore_index=True)

data = pd.concat([dfa, dfb, dfc, dfd,
                  dfe, dflove,], ignore_index=True)

X = data.drop(['sign'],  axis='columns')
y = data['sign']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

Classifier = MLPClassifier(hidden_layer_sizes=(128, 64), activation= 'relu',random_state= 42)

Classifier.fit(X_train, y_train)

y_train_pred = Classifier.predict(X_train)
y_test_pred = Classifier.predict(X_test)

print("Train Accuracy: ", accuracy_score(y_train, y_train_pred))

print("Test Accuracy: ", accuracy_score(y_test, y_test_pred))
# print(y_test)
# print(y_test_pred)

conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()



joblib.dump(Classifier, "mlp_translation_model.pkl")




# param_grid = {
#     'solver': ['adam', 'sgd', 'lbfgs'],
#     'learning_rate': ['constant', 'adaptive'],
#     'learning_rate_init': [0.001, 0.01],
#     'max_iter': [1000, 2000],
#     'batch_size': [16, 32, 64, 'auto'],
# }