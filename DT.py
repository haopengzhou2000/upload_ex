import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from sklearn import tree
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    iris = load_iris()
    data = iris["data"]
    label = iris["target"]
    feature_names = iris["feature_names"]
    target_names = iris["target_names"]
    
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.3, random_state = 22)
    
    model = DecisionTreeClassifier()
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)
    
    model.apply(X_train, check_input=True)
    model.decision_path(X_train, check_input=True)
    model.feature_importances_
    model.get_depth()
    model.get_n_leaves()
    model.get_params(deep=True)
    model.score(X_test, y_test, sample_weight=None)
    
    num_class = len(np.unique(y_train))
    evaluation_measure = {}
    accuracy = accuracy_score(y_test, pred)
    recall = recall_score(y_test, pred, average='macro') 
    precision = precision_score(y_test, pred, average='macro')
    f1_score = f1_score(y_test, pred, average='macro')
    if num_class == 2:
        auroc = roc_auc_score(y_test, pred_proba[:,1])
    else:
        auroc = roc_auc_score(y_test, pred_proba, multi_class='ovr')
    
    matplotlib.rc("font",family='MicroSoft YaHei',weight="bold")
    plt.figure(figsize=(8, 8), dpi=300)
    tree.plot_tree(model, filled=True, feature_names = feature_names, class_names = target_names)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
