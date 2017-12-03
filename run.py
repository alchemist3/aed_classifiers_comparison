from model import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load data from arff file
data, results, classes = load_arff('diabetes.arff')
X = np.array(data)
Y = np.array(results)

# Create classifiers for testing
mnb = MultinomialNB()
logistic = LogisticRegression()
neigh = KNeighborsClassifier(n_neighbors=10)
rfc = RandomForestClassifier(max_depth=2, random_state=0)
dtc = DecisionTreeClassifier(random_state=0)

classifiers = [mnb, logistic, neigh, rfc, dtc]
classifiers_names = ['MultinomialNB', 'LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier',
                     'DecisionTreeClassifier']

# Classifiers testing
metrices_names = ['dokładność', 'czułość', 'specyficzność', 'precyzja', 'F1', 'zbalansowana dokładność', 'AUC',
                  'czasu uczenia', 'czas predykcji']

characteristics = [('miary', metrices_names),
                   ('MultinomialNB', classifiers_characteristics(mnb, 50, X, Y)),
                   ('LogisticRegression', classifiers_characteristics(logistic, 50, X, Y)),
                   ('KNeighborsClassifier', classifiers_characteristics(neigh, 50, X, Y)),
                   ('RandomForestClassifier', classifiers_characteristics(rfc, 50, X, Y)),
                   ('DecisionTreeClassifier', classifiers_characteristics(dtc, 50, X, Y))]

df = pd.DataFrame.from_items(characteristics)
# pd.set_option('display.width', 1000)
print(df)

# Plot ROC for tested classifiers
roc_plots(classifiers, classifiers_names, X, Y)
