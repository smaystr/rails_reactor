import ray
from ray import tune
import torch
from torch.utils.data import DataLoader
from fnn.data_loader import ApartmentsDataset
from fnn.fnn import FNN, train_nn, validate_nn


train_dataset = ApartmentsDataset('train.csv')
test_dataset = ApartmentsDataset('test.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=100)
test_loader = DataLoader(dataset=test_dataset, batch_size=100)


def tune_nn(config):
    model = FNN(input_dim=train_dataset.get_dim(), hidden_dim=config['hidden_dim'], activation_function=config['activation_function'])
    train_nn(model=model, alpha=0.01, epochs=100, loader=train_loader)
    mae_train, mae_test, mse_train, mse_test, r2_train, r2_test = validate_nn(model=model,
                                                                                train_loader=train_loader,
                                                                                test_loader=test_loader)

    params_metrics = (f'Hyperparameters: {config}'
            f'Train MAE: {sum(mae_train)/len(mae_train)}\n'
            f'Test MAE: {sum(mae_test)/len(mae_test)}\n'
            f'Train MSE: {sum(mse_train)/len(mse_train)}\n'
            f'Test MSE: {sum(mse_test)/len(mse_test)}\n'
            f'Train R2: {sum(r2_train)/len(r2_train)}\n'
            f'Test R2: {sum(r2_test)/len(r2_test)}\n')
    
    print(params_metrics)

    tune.track.log(mae_test, mse_test, r2_test)


def main():
    model = FNN(input_dim=train_dataset.get_dim(), hidden_dim=25, hidden_num=3, activation_function='relu')
    train_nn(model=model, alpha=0.01, epochs=100, loader=train_loader)
    mae_train, mae_test, mse_train, mse_test, r2_train, r2_test = validate_nn(model=model,
                                                                                train_loader=train_loader,
                                                                                test_loader=test_loader)

    res = ('Before hyperparameters tuning:\n'
            f'Train MAE: {sum(mae_train)/len(mae_train)}\n'
            f'Test MAE: {sum(mae_test)/len(mae_test)}\n'
            f'Train MSE: {sum(mse_train)/len(mse_train)}\n'
            f'Test MSE: {sum(mse_test)/len(mse_test)}\n'
            f'Train R2: {sum(r2_train)/len(r2_train)}\n'
            f'Test R2: {sum(r2_test)/len(r2_test)}\n')
    
    print(res)

    gs = tune.run(tune_nn, config={
        'hidden_dim': tune.grid_search([10, 25, 50]),
        'activation_function': tune.grid_search(['relu', 'sigmoid', 'tanh', 'leaky', 'elu'])
    })

    print(f'Best hyperparameters: {gs.get_best_config()}')


if __name__ == '__main__':
    main()
