import paddle

"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
import sys

sys.path.append("../")
import argparse
from few_shot.datasets import fashionNet
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.maml import meta_gradient_step
from few_shot.models import FewShotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH

setup_dirs()
assert paddle.device.cuda.device_count() >= 1
device = str("cuda").replace("cuda", "gpu")
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--n", default=1, type=int)
parser.add_argument("--k", default=5, type=int)
parser.add_argument("--q", default=1, type=int)
parser.add_argument("--inner-train-steps", default=1, type=int)
parser.add_argument("--inner-val-steps", default=3, type=int)
parser.add_argument("--inner-lr", default=0.4, type=float)
parser.add_argument("--meta-lr", default=0.001, type=float)
parser.add_argument("--meta-batch-size", default=32, type=int)
parser.add_argument("--order", default=1, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--epoch-len", default=100, type=int)
parser.add_argument("--eval-batches", default=20, type=int)
args = parser.parse_args()
if args.dataset == "fashionNet":
    n_epochs = 50
    dataset_class = fashionNet
    num_input_channels = 3
    drop_lr_every = 30
else:
    raise (ValueError, "Unsupported dataset")
param_str = "{}_order={}_n={}_k={}_metabatch={}_train_steps={}_val_steps={}".format(
    args.dataset,
    args.order,
    args.n,
    args.k,
    args.meta_batch_size,
    args.inner_train_steps,
    args.inner_val_steps,
)
print(param_str)
background = dataset_class("background")
background_taskloader = paddle.io.DataLoader(
    dataset=background,
    batch_sampler=NShotTaskSampler(
        background,
        args.epoch_len,
        n=args.n,
        k=args.k,
        q=args.q,
        num_tasks=args.meta_batch_size,
    ),
    num_workers=8,
)
evaluation = dataset_class("evaluation")
evaluation_taskloader = paddle.io.DataLoader(
    dataset=evaluation,
    batch_sampler=NShotTaskSampler(
        evaluation,
        args.eval_batches,
        n=args.n,
        k=args.k,
        q=args.q,
        num_tasks=args.meta_batch_size,
    ),
    num_workers=8,
)
print("Training MAML on {}...".format(args.dataset))
meta_model = FewShotClassifier(num_input_channels, args.k, 64).to(
    device, dtype="float64"
)
meta_optimiser = paddle.optimizer.Adam(
    parameters=meta_model.parameters(), learning_rate=args.meta_lr, weight_decay=0.0
)
loss_fn = paddle.nn.CrossEntropyLoss().to(device)


def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        x = x.reshape(
            meta_batch_size, n * k + q * k, num_input_channels, x.shape[-2], x.shape[-1]
        )
        x = x.astype(dtype="float64").to(device)
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_


callbacks = [
    EvaluateFewShot(
        eval_fn=meta_gradient_step,
        num_tasks=args.eval_batches,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
    ),
    ModelCheckpoint(
        filepath=PATH + "/models/maml/{}.pth".format(param_str),
        monitor="val_{}-shot_{}-way_acc".format(args.n, args.k),
    ),
    ReduceLROnPlateau(patience=10, factor=0.5, monitor="val_loss"),
    CSVLogger(PATH + "/logs/maml/{}.csv".format(param_str)),
]
fit(
    meta_model,
    meta_optimiser,
    loss_fn,
    epochs=args.epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
    callbacks=callbacks,
    metrics=["categorical_accuracy"],
    fit_function=meta_gradient_step,
    fit_function_kwargs={
        "n_shot": args.n,
        "k_way": args.k,
        "q_queries": args.q,
        "train": True,
        "order": args.order,
        "device": device,
        "inner_train_steps": args.inner_train_steps,
        "inner_lr": args.inner_lr,
    },
)
