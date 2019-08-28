import pathlib

DB_HOSTNAME = "localhost"
DB_USERNAME = "simple_user"
DB_NAME = "apartments"
DB_PORT = 5432
TABLE_NAME = "apartments"

DB_PATH = f"postgresql://{DB_USERNAME}@{DB_HOSTNAME}:{DB_PORT}/{DB_NAME}"


CAT_FEATURES = ["offer_type", "wall_type", "heating", "city_id"]
BOOL_FEATURES = ["is_bargain", "is_used", "is_not_used", "in_installments"]
COLS_TO_REMOVE = ["title", "absolute_url", "city_name", "price_usd"]

NUM_FEATURES = [
    "position",
    "len_of_description",
    "floor_located",
    "number_of_floors_in_the_house",
    "longitude",
    "apartment_area",
    "years_elapsed",
    "num_of_punctuations_in_description",
    "number_rooms",
    "latitude",
    "num_of_uppercase_letters_in_description",
    "number_of_images_attached",
]

TARGET_COLUMN = "price_uah"

ASSETS_PATH = pathlib.Path("assets")
LGB_PATH = ASSETS_PATH / "lgb_model.jblib"
DT_PATH = ASSETS_PATH / "decision_tree.jblib"
DNN_PATH = ASSETS_PATH / "dnn.pt"
FEATURES_PATH = ASSETS_PATH / "features_train.jblib"
COLUMN_TRANSFORMER_PATH = ASSETS_PATH / "column_transformer.jblib"


REPORT_PATH = "report.md"

SAMPLE_SIZE_FOR_SPEED_INFERENCE = 1000
DNN_TRAIN_TEST_SPLIT_SIZE = 0.2

DNN_DEFAULT_FEATURES_NUM = 283
DNN_DEFAULT_HIDDEN_UNITS_DIM = 1024
DNN_LR = 1e-2
DNN_NUM_EPOCHS = 125
