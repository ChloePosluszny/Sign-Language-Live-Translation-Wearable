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
dff = pd.read_csv("training_data/f_a.csv")
dfg = pd.read_csv("training_data/g_a.csv")
dfh = pd.read_csv("training_data/h_a.csv")
dfi = pd.read_csv("training_data/i_a.csv")
dfk = pd.read_csv("training_data/k_a.csv")
dfl = pd.read_csv("training_data/l_a.csv")
dfm = pd.read_csv("training_data/m_a.csv")
dfn = pd.read_csv("training_data/n_a.csv")
dfo = pd.read_csv("training_data/o_a.csv")
dfp = pd.read_csv("training_data/p_a.csv")
dfq = pd.read_csv("training_data/q_a.csv")
dfr = pd.read_csv("training_data/r_a.csv")
dfs = pd.read_csv("training_data/s_a.csv")
dft = pd.read_csv("training_data/t_a.csv")
dfu = pd.read_csv("training_data/u_a.csv")
dfv = pd.read_csv("training_data/v_a.csv")
dfw = pd.read_csv("training_data/w_a.csv")
dfx = pd.read_csv("training_data/x_a.csv")
dfy = pd.read_csv("training_data/y_a.csv")

dfa_d = pd.read_csv('training_data/a_d.csv')
dfb_d = pd.read_csv("training_data/b_d.csv")
dfc_d = pd.read_csv("training_data/c_d.csv")
dfd_d = pd.read_csv("training_data/d_d.csv")
dfe_d = pd.read_csv("training_data/e_d.csv")
dflove_d = pd.read_csv("training_data/i love you_d.csv")
dff_d = pd.read_csv("training_data/f_d.csv")
dfg_d = pd.read_csv("training_data/g_d.csv")
dfh_d = pd.read_csv("training_data/h_d.csv")
dfi_d = pd.read_csv("training_data/i_d.csv")
dfk_d = pd.read_csv("training_data/k_d.csv")
dfl_d = pd.read_csv("training_data/l_d.csv")
dfm_d = pd.read_csv("training_data/m_d.csv")
dfn_d = pd.read_csv("training_data/n_d.csv")
dfo_d = pd.read_csv("training_data/o_d.csv")
dfp_d = pd.read_csv("training_data/p_d.csv")
dfq_d = pd.read_csv("training_data/q_d.csv")
dfr_d = pd.read_csv("training_data/r_d.csv")
dfs_d = pd.read_csv("training_data/s_d.csv")
dft_d = pd.read_csv("training_data/t_d.csv")
dfu_d = pd.read_csv("training_data/u_d.csv")
dfv_d = pd.read_csv("training_data/v_d.csv")
dfw_d = pd.read_csv("training_data/w_d.csv")
dfx_d = pd.read_csv("training_data/x_d.csv")
dfy_d = pd.read_csv("training_data/y_d.csv")


data = pd.concat([
    dfa, dfb, dfc, dfd, dflove, dfe, dff, dfg, dfh, dfi, dfk, dfl, dfm, dfn, dfo,
    dfp, dfq, dfr, dfs, dft, dfu, dfv, dfw, dfx, dfy,
    dfa_d, dfb_d, dfc_d, dfd_d, dflove_d, dfe_d, dff_d, dfg_d, dfh_d, dfi_d, dfk_d, dfl_d, 
    dfm_d, dfn_d, dfo_d, dfp_d, dfq_d, dfr_d, dfs_d, dft_d, dfu_d, dfv_d, dfw_d, dfx_d, dfy_d
], ignore_index=True)

# data = pd.concat([dfa, dfb, dfc, dfd,
#                   dfe, dflove,], ignore_index=True)
# print(data)

X = data.drop(['sign'],  axis='columns')
y = data['sign']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

Classifier = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation= 'relu',random_state= 42)

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