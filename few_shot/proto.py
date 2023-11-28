import paddle
from typing import Callable
import pdb
from few_shot.utils import pairwise_distances


def proto_net_episode(
    model: paddle.nn.Layer,
    optimiser: paddle.optimizer.Optimizer,
    loss_fn: Callable,
    x: paddle.Tensor,
    y: paddle.Tensor,
    n_shot: int,
    k_way: int,
    q_queries: int,
    distance: str,
    train: bool,
):
    """Performs a single training episode for a Prototypical Network.

    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        model.train()
        optimiser.clear_grad()
    else:
        model.eval()
    embeddings = model(x)
    support = embeddings[: n_shot * k_way]
    queries = embeddings[n_shot * k_way :]
    prototypes = compute_prototypes(support, k_way, n_shot)
    distances = pairwise_distances(queries, prototypes, distance)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)
    y_pred = paddle.nn.functional.softmax(-distances, axis=1)
    if train:
        loss.backward()
        optimiser.step()
    else:
        pass
    return loss, y_pred


def compute_prototypes(support: paddle.Tensor, k: int, n: int) -> paddle.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: class_prototypestypes aka mean embeddings for each class
    """
    class_prototypes = support.reshape(k, n, -1).mean(axis=1)
    return class_prototypes
