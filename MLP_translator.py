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

dfa = pd.read_csv('training_data/a_a.csv')
dfb = pd.read_csv("training_data/b_a.csv")
dfc = pd.read_csv("training_data/c_a.csv")
dfd = pd.read_csv("training_data/d_a.csv")
dfe = pd.read_csv("training_data/e_a.csv")
dflove = pd.read_csv("training_data/i love you_a.csv")
dff = pd.read_csv("f_a.csv")
dfg = pd.read_csv("g_a.csv")
dfh = pd.read_csv("h_a.csv")
dfi = pd.read_csv("i_a.csv")
#dfj = pd.read_csv("j_a.csv")
dfk = pd.read_csv("k_a.csv")
dfl = pd.read_csv("l_a.csv")
dfm = pd.read_csv("m_a.csv")
dfn = pd.read_csv("n_a.csv")
dfo = pd.read_csv("o_a.csv")
dfp = pd.read_csv("p_a.csv")
dfq = pd.read_csv("q_a.csv")
dfr = pd.read_csv("r_a.csv")
dfs = pd.read_csv("s_a.csv")
dft = pd.read_csv("t_a.csv")
dfu = pd.read_csv("u_a.csv")
dfv = pd.read_csv("v_a.csv")
dfw = pd.read_csv("w_a.csv")
dfx = pd.read_csv("x_a.csv")
dfy = pd.read_csv("y_a.csv")
#dfz = pd.read_csv("z_a.csv")

# data = pd.concat([dfa, dfb, dfc, dfd,
#                   dfe, dff, dfg, dfh, dfi, dfj, dfk, dfl, dfm, dfn, dfo,
#                     dfp, dfq, dfr, dfs, dft, dfu, dfv, dfw, dfx, dfy, dfz,], ignore_index=True)

data = pd.concat([dfa, dfb, dfc, dfd,
                  dfe, dflove,], ignore_index=True)
print(data)

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