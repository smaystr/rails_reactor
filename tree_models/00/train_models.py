import lightgbm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from operator import itemgetter

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from my_utils import *
from functools import partial
from HA6.linear_models_high_level import *
import my_config

from dnn import PriceRegressorDNN


def profile_model(model, features, target, serialize_path):

    cv_func = partial(
        cross_val_score, model, features, target, cv=5, verbose=2, n_jobs=-1
    )

    r_2_score = np.mean(cv_func())

    # np.abs is needed to adjust 'is_greater_better' strategy, which gives a negative sign to the score
    rmse_score = np.mean(np.abs(cv_func(scoring=rmse_scorer)))

    time_start = time.time()
    model.fit(features, target)
    time_elapsed_train = time.time() - time_start

    inference_sample = features[
        np.random.randint(
            0, len(features), size=my_config.SAMPLE_SIZE_FOR_SPEED_INFERENCE
        )
    ]
    time_start = time.time()
    model.predict(inference_sample)
    time_elapsed_inference = time.time() - time_start

    score_dict = {
        "cv_explained_variance": r_2_score,
        "cv_rmse": rmse_score,
        "training_time": time_elapsed_train,
        f"inference_time_for_{my_config.SAMPLE_SIZE_FOR_SPEED_INFERENCE}": time_elapsed_inference,
    }

    write_stats_to_file(model, score_dict)

    joblib.dump(model, serialize_path)


def serialize_features(data):
    features_train = list(set(data) - {my_config.TARGET_COLUMN})
    joblib.dump(features_train, f"{my_config.FEATURES_PATH}")


def train_models():

    data = load_data()
    data_with_needed_cols = preprocess_columns(data)

    serialize_features(data_with_needed_cols)

    features, target = prepare_data(data_with_needed_cols)

    train_dnn(features, target)
    train_models_sklearn_api(features, target)


def train_models_sklearn_api(features, target):

    lgb_reg = lightgbm.LGBMRegressor(
        max_depth=30,
        num_leaves=128,
        learning_rate=0.05,
        n_estimators=500,
        bagging_fraction=0.6,
        feature_fraction=0.6,
        random_state=42,
    )

    tree = DecisionTreeRegressor(max_depth=10, random_state=42)

    profile_model(lgb_reg, features, target, my_config.LGB_PATH)

    profile_model(tree, features, target, my_config.DT_PATH)


def train_dnn(features, target):

    # print(features, target)
    torch.manual_seed(42)

    X_train, X_test, Y_train, Y_test = train_test_split(
        features, target, test_size=my_config.DNN_TRAIN_TEST_SPLIT_SIZE
    )

    print(X_train.shape)
    print(X_test.shape)
    # print(X_train.shape)
    # print(Y_train.shape)

    X_train, X_test, Y_train, Y_test = map(
        lambda x: torch.from_numpy(x).float(), [X_train, X_test, Y_train, Y_test]
    )

    train_loader, test_loader = create_loaders(X_train, X_test, Y_train, Y_test)

    model_dnn = PriceRegressorDNN(
        features.shape[1],
        num_hidden_units=my_config.DNN_DEFAULT_HIDDEN_UNITS_DIM,
        activation_function=nn.ReLU(),
    )

    criterion = nn.L1Loss()

    optimizer = torch.optim.SGD(
        model_dnn.parameters(), lr=my_config.DNN_LR, momentum=0.9
    )

    training_stats = train_model(
        model_dnn,
        optimizer,
        criterion,
        train_loader,
        my_config.DNN_NUM_EPOCHS,
        torch_rmse,
        print_epoch=10,
    )

    metric_vals = list(
        map(
            itemgetter(1),
            [
                test_model(model_dnn, test_loader, metric)
                for metric in [torch_rmse, torch_r2_score]
            ],
        )
    )

    write_stats_to_file(
        model_dnn,
        {
            "test_rmse": metric_vals[0],
            "test_explained_variance": metric_vals[1],
            "train_test_split_size": my_config.DNN_TRAIN_TEST_SPLIT_SIZE,
        },
    )

    generate_report(
        training_stats, report_path=my_config.REPORT_PATH, figure_path="./my_figures"
    )

    torch.save(model_dnn.state_dict(), my_config.DNN_PATH)


if __name__ == "__main__":
    train_models()
