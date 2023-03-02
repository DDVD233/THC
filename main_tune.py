from pathlib import Path
import argparse
from timeit import repeat
import yaml
import torch
from models import GraphTransformer, THC
from train_tune import TuneTrain
from datetime import datetime
from dataloader import init_dataloader
import wandb
from torch import nn
import ray
from ray import tune, air
from ray.air.checkpoint import Checkpoint
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune.search.bayesopt import BayesOptSearch
import numpy as np
from pathlib import Path
import os
from ray.tune.search.hyperopt import HyperOptSearch


def main(config, data_dir):
    data = config['data']
    for key, value in data.items():
        if isinstance(value, str) and 'dataset' in value:
            data[key] = data_dir + value
    dataloaders, node_size, node_feature_size, timeseries_size = \
        init_dataloader(config['data'])

    config['train']["seq_len"] = timeseries_size
    config['train']["node_size"] = node_size

    model = THC(config['model'], node_num=node_size, lr=config['train']['lr'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    use_train = TuneTrain

    if config['train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'])
    elif config['train']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'])
    else:
        raise ValueError(f"Optimizer {config['train']['optimizer']} not supported")
    opts = (optimizer,)

    loss_name = 'loss'

    # wandb.config = {
    #     "learning_rate": config['train']['lr'],
    #     "epochs": config['train']['epochs'],
    #     "batch_size": config['data']['batch_size'],
    #     "dataset": config['data']["dataset"],
    #     "model": config['model']['type']
    # }


    now = datetime.now()

    date_time = now.strftime("%m-%d-%H-%M-%S")

    # wandb.run.name = f"{date_time}_{config['data']['dataset']}_{config['model']['type']}"

    extractor_type = config['model']['extractor_type'] if 'extractor_type' in config['model'] else "none"
    embedding_size = config['model']['embedding_size'] if 'embedding_size' in config['model'] else "none"
    window_size = config['model']['window_size'] if 'window_size' in config['model'] else "none"

    if "graph_generation" in config['model'] and config['model']["graph_generation"]:
        model_name = f"{config['train']['method']}_{config['model']['graph_generation']}"
    else:
        model_name = f"{config['train']['method']}"

    save_folder_name = Path(config['train']['log_folder'])/Path(
        date_time +
        f"_{config['data']['dataset']}_{config['model']['type']}_{model_name}"
        + f"_{extractor_type}_{loss_name}_{embedding_size}_{window_size}")


    train_process = use_train(
        config, model, opts, dataloaders, save_folder_name, session=session)

    train_process.train()


def sample_size():
    first_layer = np.random.randint(4, 90)
    second_layer = np.random.randint(1, int(first_layer/2))
    result = [first_layer, second_layer]
    # if np.random.random() > 0.5 and second_layer > 10:
    #     third_layer = np.random.randint(1, int(second_layer/2))
    #     result.append(third_layer)
    return result

def tune_model(base_config: str):
    ray.init()
    with open(base_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        train_config = config['train']
        train_config['lr'] = tune.loguniform(1e-6, 1e-2)
        train_config['weight_decay'] = tune.loguniform(1e-6, 1e-2)
        train_config['optimizer'] = tune.choice(['adam', 'sgd'])
        train_config["wandb"] = {"project": "BrainTransformer2", "entity": "ddavid233"}
        if "abide" in base_config:
            train_config["wandb"]["dataset"] = "abide"
        elif "abcd" in base_config:
            train_config["wandb"]["dataset"] = "abcd"

        config['train'] = train_config

        model_config = config['model']
        # model_config['sizes'] = tune.sample_from(sample_size)
        model_config['sizes'] = list(model_config['sizes'])
        model_config['orthogonal'] = tune.choice([True, False])
        model_config['readout'] = tune.choice(['mean', 'sum', 'max', 'concat'])
        model_config['att_noise'] = tune.choice([True, False])
        model_config["encoder_hidden_size"] = tune.choice([128, 256, 512, 1024, 2048, 4096])
        model_config["transformer_hidden_size"] = tune.choice([128, 256, 512, 1024, 2048, 4096])
        model_config["readout_hidden_size"] = tune.choice([64, 128, 256, 512, 1024, 2048])
        model_config["feature_reduction_dim"] = tune.choice([1, 2, 4, 8, 16, 32])
        model_config["mask_top_k"] = tune.choice([True, False])

        scheduler = ASHAScheduler(
            max_t=train_config["epochs"],
            metric='Val AUC',
            mode='max',
            grace_period=10,
            reduction_factor=2)
        generator = HyperOptSearch()
        repeater = tune.search.Repeater(searcher=generator, repeat=5)
        # Current script directory, absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(main, data_dir=current_dir),
                resources={"cpu": 4, "gpu": 0.5}
            ),
            tune_config=tune.TuneConfig(
                num_samples=-1,
                search_alg=repeater,
                metric="Val AUC",
                mode="max",
            ),
            param_space=config,
        )
        results = tuner.fit()

        best_result = results.get_best_result("Val AUC", "max")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
            best_result.metrics["Test Loss"]))
        print("Best trial final validation auc: {}".format(
            best_result.metrics["Val AUC"]))

        # test_best_model(best_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/pnc.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=5, type=int)
    parser.add_argument('--wandb', default="ddavid233", type=str)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    tune_model(args.config_filename)
