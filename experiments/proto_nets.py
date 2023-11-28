import paddle

"""
Implementation of fashionNet results of Snell et al Prototypical networks.

"""
import sys

sys.path.append("../")
import argparse
from few_shot.datasets import fashionNet
from few_shot.models import get_few_shot_encoder, resnet_pretrained
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH

setup_dirs()
assert paddle.device.cuda.device_count() >= 1
device = str("cuda").replace("cuda", "gpu")
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="fashionNet")
parser.add_argument("--distance", default="cosine")
parser.add_argument("--n-train", default=1, type=int)
parser.add_argument("--n-test", default=1, type=int)
parser.add_argument("--k-train", default=60, type=int)
parser.add_argument("--k-test", default=5, type=int)
parser.add_argument("--q-train", default=5, type=int)
parser.add_argument("--q-test", default=1, type=int)
args = parser.parse_args()
evaluation_episodes = 100
episodes_per_epoch = 10
if args.dataset == "fashionNet":
    n_epochs = 50
    dataset_class = fashionNet
    num_input_channels = 3
    drop_lr_every = 30
else:
    raise (ValueError, "Unsupported dataset")
param_str = "{}_nt={}_kt={}_qt={}_nv={}_kv={}_qv={}".format(
    args.dataset,
    args.n_train,
    args.k_train,
    args.q_train,
    args.q_train,
    args.n_test,
    args.k_test,
    args.q_test,
)
print(param_str)
background = dataset_class("background")
background_taskloader = paddle.io.DataLoader(
    dataset=background,
    batch_sampler=NShotTaskSampler(
        background, episodes_per_epoch, args.n_train, args.k_train, args.q_train
    ),
    num_workers=4,
)
evaluation = dataset_class("evaluation")
evaluation_taskloader = paddle.io.DataLoader(
    dataset=evaluation,
    batch_sampler=NShotTaskSampler(
        evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test
    ),
    num_workers=4,
)
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype="float64")
print("Training Prototypical network on {}...".format(args.dataset))
optimiser = paddle.optimizer.Adam(
    parameters=model.parameters(), learning_rate=0.001, weight_decay=0.0
)
loss_fn = paddle.nn.NLLLoss()


def lr_schedule(epoch, lr):
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance,
    ),
    ModelCheckpoint(
        filepath=PATH + "/models/proto_nets/{}.pth".format(param_str),
        monitor="val_{}-shot_{}-way_acc".format(args.n_test, args.k_test),
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + "/logs/proto_nets/{}.csv".format(param_str)),
]
fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=["categorical_accuracy"],
    fit_function=proto_net_episode,
    fit_function_kwargs={
        "n_shot": args.n_train,
        "k_way": args.k_train,
        "q_queries": args.q_train,
        "train": True,
        "distance": args.distance,
    },
)
