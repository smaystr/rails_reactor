from linear_regressor import MyLinearRegression
from logistic_regressor import MyLogisticRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from utils import preprocess_data


def test_logistic_regressor():

    print("\nTesting custom LogReg\n")

    cat_columns_heart_df = ["sex", "exang", "slope", "fbs", "cp", "restecg", "thal"]
    X_train, X_test, Y_train, Y_test = preprocess_data("heart", cat_columns_heart_df)

    lr = MyLogisticRegression(
        learning_rate=0.05, num_iterations=5000, C=0.1, print_steps=200
    )

    lr_sk = LogisticRegression(C=0.1, max_iter=5000, solver="liblinear")

    print("Fitting custom log reg\n")
    lr.fit(X_train, Y_train)

    print("Fitting sklearn log reg\n")
    lr_sk.fit(X_train, Y_train.flatten())

    print(
        f"Custom model accuracy {lr.score(X_test, Y_test)}\n"
        f"Sklearn model accuracy {lr_sk.score(X_test, Y_test)}"
    )


def test_linear_regressor():

    print("\nTesting custom LinReg\n")

    cat_columns_insurance = ["region", "sex", "smoker"]
    X_train, X_test, Y_train, Y_test = preprocess_data(
        "insurance", cat_columns_insurance, target_column="charges"
    )
    lr = MyLinearRegression(learning_rate=1e-3, num_iterations=50000, lam=0)

    lr_sk = LinearRegression()

    print("Fitting custom linear reg\n")
    lr.fit(X_train, Y_train)

    print("Fitting sklearn linear reg\n")
    lr_sk.fit(X_train, Y_train)

    print(
        f"Custom model R^2 coef {lr.score(X_test, Y_test)}\n"
        f"Sklearn model R^2  coef {lr_sk.score(X_test, Y_test)}"
    )


def main():

    test_linear_regressor()
    test_logistic_regressor()


if __name__ == "__main__":
    main()
