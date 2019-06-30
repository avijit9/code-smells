# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def main():
    df = pd.read_csv('datasets.csv')
    smells = ['OSE', 'BCE', 'PDE', 'SV', 'OS', 'SDS', 'RS', 'TFS']

    m = Models()
    for smell in smells:
        print('### Predicting code smell "{}" ###'.format(smell))
        m.predict(df, smell)


class Models():

    def predict(self, df, feature):
        # train test split
        X = df.iloc[:, 1:24]
        y = df = df[[feature]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)

        # Naive Bayes
        print('-- Naive Bayes --')
        clf = GaussianNB()
        self.output_accuracy(X, y, X_train, y_train, X_test, y_test, clf)

        # Random Forests
        print('-- Random Forests --')
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=2, random_state=0)
        self.output_accuracy(X, y, X_train, y_train, X_test, y_test, clf)

        # C4.5 (J48)
        print('-- C4.5 (implented as J48 in Weka) --')
        clf = DecisionTreeClassifier(random_state=0)
        self.output_accuracy(X, y, X_train, y_train, X_test, y_test, clf)

        # Support Vector Machine using LIBSVM implementation with SMO
        print('-- Support Vector Machine using LIBSVM implementation with SMO --')
        clf = SVC(gamma='auto')
        self.output_accuracy(X, y, X_train, y_train, X_test, y_test, clf)

    def output_accuracy(self, X, y, X_train, y_train, X_test, y_test, clf):
        clf.fit(X_train, y_train.values.ravel())
        train_acc = clf.score(X_train, y_train.values.ravel())
        test_acc = clf.score(X_test, y_test.values.ravel())
        cv_scores = cross_val_score(clf, X, y.values.ravel(), cv=5)
        cv_acc = '{} (+/- {})'.format(cv_scores.mean(), cv_scores.std() * 2)
        print('Training accuray: {}'.format(train_acc))
        print('Testing accuray: {}'.format(test_acc))
        print('Cross-validation accuray: {}'.format(cv_acc))
        print('\n')


if __name__ == '__main__':
    main()


# %%
