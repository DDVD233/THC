import torch
from torch.nn import Module
from models import THC
from models.model_diffpool import DiffPool
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def get_attention_scores(model: THC, inputs) -> [torch.Tensor]:
    model.eval()
    with torch.no_grad():
        # Because we are just getting the attention scores, there's probably no need to move the model to GPU.
        model(inputs)
        attention_scores = model.get_attention_weights()
    return attention_scores


def load_model_and_get_attention_scores(model_path: str, config: dict, inputs):
    model_type = config['model']['type']
    if model_type == 'dec_transformer':
        model: THC = THC(model_config=config['model'],
                         node_num=config['train']["node_size"])
    model = load_model(model_path, model)
    return get_attention_scores(model, inputs)


def get_assignments(model: THC, inputs) -> [torch.Tensor]:
    # model.eval()
    _, assignment = model(inputs)
    return assignment


def get_assignments_with_loader(model: THC, loader) -> [torch.Tensor]:
    # model.eval()
    assignments = []
    count = 0
    for pearson, data_in, label in loader:
        _, assignment = model(data_in)
        if len(assignments) == 0:
            assignments = assignment
        else:
            for layer in range(len(assignment)):
                assignments[layer] += assignment[layer]
        count += 1
    for layer in range(len(assignments)):
        assignments[layer] /= count
    return assignments


def load_model_and_get_assignments(model_path: str, config: dict, inputs):
    if config['model']['type'] == 'diffpool':
        model: DiffPool = DiffPool(model_config=config['model'],
                                               node_num=config['train']["node_size"])
    else:
        model: THC = THC(model_config=config['model'],
                         node_num=config['train']["node_size"])
    model = load_model(model_path, model)
    return get_assignments(model, inputs)


def load_model_and_get_assignments_with_loader(model_path: str, config: dict, loader):
    model: THC = THC(model_config=config['model'],
                     node_num=config['train']["node_size"])
    model = load_model(model_path, model)
    return get_assignments_with_loader(model, loader)


def load_model_and_get_cluster_centers(model_path: str, config: dict):
    model: THC = THC(model_config=config['model'],
                     node_num=config['train']["node_size"])
    model = load_model(model_path, model)
    return model.get_cluster_centers()
