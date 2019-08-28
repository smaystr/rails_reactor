import time

from torch.utils.data import DataLoader

from hw4.cross_validation import train_test_split
from hw4.metrics import *
from hw4.utilities import set_up_logging, read_file
from hw6.models_torch import train_model, LogisticRegression, LinearRegression
from hw6.models_torch_data import MyDataset
from hw6.utilities import parse_args

if __name__ == '__main__':
    args = parse_args()

    random_state = 42
    set_up_logging(args.log, args.verbose)

    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('gpu')
    else:
        device = torch.device('cpu')

    X, y, columns = read_file(args.dataset, args.target, args.na, args.categorical)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    train_dataset = MyDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)

    test_dataset = MyDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    if args.task == 'logistic':
        model = LogisticRegression(train_dataset.get_dim(), 1, device)
        criterion = torch.nn.BCELoss()
    else:
        model = LinearRegression(train_dataset.get_dim(), 1, device)
        criterion = torch.nn.MSELoss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    start_time = time.time()
    train_model(model, optimizer, criterion, train_loader, args.epochs, device)
    fit_time = time.time() - start_time

    preds = []
    trues = []
    with torch.no_grad():
        for data in test_loader:
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)

            pred = model(X_test)

            pred = torch.tensor([i[0] for i in pred], dtype=torch.float32)
            y_test = torch.tensor([i[0] for i in y_test], dtype=torch.float32)

            if args.batch_size > 1:
                preds.extend(pred)
                trues.extend(y_test)
            else:
                preds.append(pred)
                trues.append(y_test)

    trues = torch.tensor(trues, dtype=torch.float32)
    preds = torch.tensor(preds, dtype=torch.float32)

    if args.task == 'logistic':
        loss = log_loss(trues, preds)
        preds = torch.tensor([p.round() for p in preds], dtype=torch.float32)

        m = {
            'accuracy': accuracy(trues, preds),
            'recall': recall(trues, preds),
            'precision': precision(trues, preds),
            'f1': f1(trues, preds),
            'log-loss': loss
        }
    else:
        m = {
            'mse': mse(trues, preds),
            'rmse': rmse(trues, preds),
            'mae': mae(trues, preds),
            'mape': mape(trues, preds),
            'mpe': mpe(trues, preds),
            'r2': r2(trues, preds)
        }
    metrics_out = '\n'.join(f'               {k}: {v}' for k, v in m.items())
    print(f'Model metrics: \n{metrics_out}')
    print(f'Fit time:\n          {fit_time} s')
