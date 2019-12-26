import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
# from torchsummary import summary

import torch
import torch.backends.cudnn # Backend for using NVIDIA CUDA
import numpy as np
import pickle

from torch import nn, optim
from torch.nn import functional as F
from torch.optim .optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils import data

import argparse
from pathlib import Path

# Enable benchmark mode on CUDNN since the input sizes do not vary. This finds the best algorithm to implement the convolutions given the layout.
torch.backends.cudnn.benchmark = True

# Add argument parser
parser = argparse.ArgumentParser(
    description="Train a CNN on UrbanSound8K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Size of mini-batches for SGD",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=5,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=300,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
)
parser.add_argument(
    "--dropout",
    default=0.5,
    type=float,
)
parser.add_argument(
    "--mode",
    default="LMC",
    type=str,
    help="The type of data to train the network on (LMC, MC, MLMC)"
)

class SoundShape(NamedTuple): # 45*85*1
    height: int
    width: int
    channels: int

# Use GPU if cuda is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA...")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU...")

# The Dataset class
class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):

        dataset = np.array(self.dataset)

        LM = dataset[index]["features"]["logmelspec"]
        MFCC = dataset[index]["features"]["mfcc"]
        C = dataset[index]["features"]["chroma"]
        SC = dataset[index]["features"]["spectral_contrast"]
        T = dataset[index]["features"]["tonnetz"]

        if self.mode == 'LMC':
            # Edit here to load and concatenate the neccessary features to
            # create the LMC feature
            LMC = np.concatenate((LM, C, SC, T), axis=0)
            feature = torch.from_numpy(LMC.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to
            # create the MC feature
            MC = np.concatenate((MFCC, C, SC, T), axis=0)
            feature = torch.from_numpy(MC.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to
            # create the MLMC feature
            MLMC = np.concatenate((MFCC, LM, C, SC, T), axis=0)
            feature = torch.from_numpy(MLMC.astype(np.float32)).unsqueeze(0)

        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label

    def __len__(self):
        return len(self.dataset)

# The model class
class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels:int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = SoundShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3,3),
            padding=(1,1),
        )
        self.initialise_layer(self.conv1)
        self.bnorm1 = nn.BatchNorm2d(
            num_features=32
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3,3),
            padding=(1,1),
        )
        self.initialise_layer(self.conv2)

        self.bnorm2 = nn.BatchNorm2d(
            num_features=32
        )
        # Pooling Layer with stride to half the output
        self.pool2 = nn.MaxPool2d(
            kernel_size=(2,2),
            stride=(2,2),
            padding=(1,1),
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3,3),
            padding=(1,1),
        )
        self.bnorm3 = nn.BatchNorm2d(
            num_features=64
        )

        # Could use Max Pooling for the last layer, but probably more likely to be stride
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            padding=(1,1),
            stride=(2,2),
        )
        self.bnorm4 = nn.BatchNorm2d(
            num_features=64
        )

        self.fc1 = nn.Linear(15488, 1024)
        self.initialise_layer(self.fc1)
        self.bnormfc1 = nn.BatchNorm1d(
            num_features = 1024
        )

        self.fcout = nn.Linear (1024, 10)
        self.initialise_layer(self.fcout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sounds: torch.Tensor) -> torch.Tensor:
        # Hidden Layer 1
        x = self.conv1(sounds)
        x = self.bnorm1(x)
        x = F.relu(x)

        # Hidden Layer 2
        x = self.conv2(self.dropout(x))
        x = self.bnorm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Hidden Layer 3
        x = self.conv3(x)
        x = self.bnorm3(x)
        x = F.relu(x)

        # Hidden Layer 4
        x = self.conv4(self.dropout(x))
        x = self.bnorm4(x)
        x = F.relu(x)

        # Flatten Hidden Layer 4
        x = torch.flatten(x, start_dim = 1)

        # Fully Conected Layer 1 (do we put dropout on both FC layers?)
        x = self.fc1(self.dropout(x))
        x = self.bnormfc1(x) # This was not in the paper
        x = torch.sigmoid(x)

        # Fully Conected Layer 2 (do we put dropout on both FC layers?)
        x = self.fcout(x)

        return x



    # Initialise weights using Kaiming
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                # Compute the forward pass
                logits = self.model.forward(batch)

                # Calculate the loss
                loss = self.criterion(logits, labels)

                # Compute backpropogation
                loss.backward()

                # Update the SGD optimiser parameters and set the update grads to zero again
                self.optimizer.step()
                self.optimizer.zero_grad()

                # disabling autograd when calculationg the accuracy
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                # Writing to logs and printing out the progress
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                # Update loop params for next batch
                self.step += 1
                data_load_start_time = time.time()

            # Write to summary writer at the end of each epoch
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()

    # Printing logs
    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f},"

        )

    # Writing logs
    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        class_accuracy = compute_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )

        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}, class_accuracy: {class_accuracy}")
        self.model.train()

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

def compute_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray], class_count: int = 10) -> float:
    assert len(labels) == len(preds)
    class_accuracy = []
    for class_label in range(0,class_count):
        class_labels = np.where(labels == class_label, class_label, class_label)
        # twos = 2*np.ones(class_labels.shape)
        class_accuracy.append(float(np.logical_and((preds == class_labels),(labels == class_labels)).sum())*100 / np.array(labels == class_labels).sum())
    return class_accuracy


def main(args):

    # Setup directory for the logs
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    # Load and prepare the data
    train_dataset = UrbanSound8KDataset("./UrbanSound8K_train.pkl", args.mode)
    test_dataset = UrbanSound8KDataset("./UrbanSound8K_test.pkl", args.mode)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    # Create the CNN model
    model = CNN(height=41, width=85, channels=1, class_count=10, dropout=args.dropout)

    # summary(model, (1,41, 85))

    # Define the loss function criterion (softmax cross entropy)
    criterion = nn.CrossEntropyLoss()

    # Defining the SGD optimised used
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True)

    # Trying with adam optimiser instead
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))

    # Train the model
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE)
    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

# Log directory management
def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (f'CNN_bn_epochs={args.epochs}_dropout={args.dropout}_bs={args.batch_size}_lr={args.learning_rate}_momentum={args.momentum}_mode={args.mode}_run_')
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    start = time.time()
    main(parser.parse_args())
    print("Total time taken: {}".format(time.time() - start))
