from pathlib import Path
import argparse
from timeit import repeat
import yaml
import torch
from models import GraphTransformer, THC
from train import BrainGNNTrain
from datetime import datetime
from dataloader import init_dataloader
import wandb


def main(args):
   
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

        dataloaders, node_size, node_feature_size, timeseries_size = \
            init_dataloader(config['data'])

        config['train']["seq_len"] = timeseries_size
        config['train']["node_size"] = node_size
        if config['model']['type'] == 'transformer':
            model = GraphTransformer(config['model'], node_num=node_size)
        elif config['model']['type'] == 'dec_transformer':
            model = THC(config['model'], node_num=node_size, lr=config['train']['lr'])
        elif config['model']['type'] == 'diffpool':
            from models.model_diffpool import DiffPool
            model = DiffPool(config['model'], node_num=node_size)

        use_train = BrainGNNTrain

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'])
        opts = (optimizer,)

        loss_name = 'loss'


        wandb.config = {
            "learning_rate": config['train']['lr'],
            "epochs": config['train']['epochs'],
            "batch_size": config['data']['batch_size'],
            "dataset": config['data']["dataset"],
            "model": config['model']['type']
        }


        now = datetime.now()

        date_time = now.strftime("%m-%d-%H-%M-%S")

        wandb.run.name = f"{date_time}_{config['data']['dataset']}_{config['model']['type']}"

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
            config['train'], model, opts, dataloaders, save_folder_name)

        train_process.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/pnc.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=5, type=int)
    parser.add_argument('--wandb', default="ddavid233", type=str)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    yaml_name = Path(args.config_filename).name
    dataset_name = yaml_name.split("_")[0]
    tags = [f"{dataset_name}_project"]
    other_tags = yaml_name.split(".")[0].split("_")
    tags.extend(other_tags)

    for i in range(args.repeat_time):
        run = wandb.init(project="BrainTransformer2", entity=args.wandb, reinit=True,
                         group=yaml_name, tags=tags)
        main(args)
        run.finish()
