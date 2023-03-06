from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb


def train_loop(dataloader, model, loss_fn, optimizer):
    data_size = len(dataloader.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{data_size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = ClassificationLoss()
    loss_fn.to(device)

    learning_rate = 0.002
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95, dampening=1e-5,
                                weight_decay=1e-5)

    train_trans = torchvision.transforms.Compose(
        (
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32),
            torchvision.transforms.ColorJitter(0.7, 0.2),
            torchvision.transforms.ToTensor()
        )
    )
    val_trans = torchvision.transforms.Compose(
        (
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.CenterCrop(size=32),
            torchvision.transforms.ToTensor()
        )
    )

    train_loader = load_data('data/train', transform=train_trans)
    val_loader = load_data('data/valid', transform=val_trans)

    epochs = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(val_loader, model, loss_fn)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
