import torch
import pandas as pd
import dbaccess

from sklearn.model_selection import train_test_split
from model.preprocessing import transform_new_data, load_data, load_and_transform_database


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        y_pred = self.linear3(h2_relu)

        return y_pred


def fit(model, X_train, y_train, num_epochs=5000):
    losses = []

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(num_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_model(d_in=33, h=1000, d_out=1, num_epochs=5000, test_size=0.2, random_state=0, output_model_path='model_new.pt'):
    conn = dbaccess.open_db()

    db = load_and_transform_database(conn)

    X = db.drop(columns='price_usd')
    Y = db['price_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = load_data(X_train, X_test, y_train, y_test, 'cpu')
    print('------1.5---------')
    model = TwoLayerNet(d_in, h, d_out)
    fit(model, X_train, y_train, num_epochs)
    torch.save(model, output_model_path)
    return torch.sqrt(torch.nn.MSELoss()(model(X_test), y_test))


def predict_price(params, path_model='model_new.pt', db_path='transformed_db.csv'):
    model = torch.load(path_model)
    db = pd.read_csv(db_path, index_col=['item_id'])
    tr_features, true_price  = transform_new_data(params, db)
    return int(model(torch.tensor(tr_features, dtype=torch.float32))), true_price
