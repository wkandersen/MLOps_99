from model import CustomResNet50, SimpleCNN
from data import load_data
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import hydra
from torchvision.models import resnet50
from torchvision import models


# loading
cuda = torch.cuda.is_available()
@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")

def train(config):
    hparams = config['hyperparameters']

    # model = SimpleCNN(num_classes=hparams['num_classes'], x_dim = hparams['x_dim'])
    model = CustomResNet50(num_classes=hparams['num_classes'], weights=config["weights"], x_dim=hparams['x_dim'], dropout_rate=config["dropout_rate"])
    optimizer = Adam(model.parameters(), hparams["lr"])
    criterion = CrossEntropyLoss()

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    train_loader, valid_loader, train_subset_new = load_data()

    for epoch in range(hparams['epochs']):
        # Training

        # Model, optimizer, criterion initialization
        # model = CustomResNet50(num_classes=hparams['num_classes'], pretrained=True)


        # Load data

        #create subset of dataa

        # Training loop
        model.train()
        correct = 0
        train_loss = 0.0
        tbar = tqdm(train_loader, desc = 'Training', position=0, leave=True)
        for i,(inp,lbl) in enumerate(tbar):
            optimizer.zero_grad()


            if cuda:
                inp,lbl = inp.cuda(),lbl.cuda()

                
            out = model(inp)
            loss = criterion(out,lbl)
            train_loss += loss
            out = out.argmax(dim=1)
            correct += (out == lbl).sum().item()
            loss.backward()
            optimizer.step()
            tbar.set_description(f"Epoch: {epoch+1}, loss: {loss.item():.5f}, acc: {100.0*correct/((i+1)*train_loader.batch_size):.4f}%")
        train_acc = 100.0*correct/len(train_loader.dataset)
        train_loss /= (len(train_loader.dataset)/hparams['batch_size'])

        print("Training complete")
        torch.save(model.state_dict(), "models/model.pth")
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # axs[0].plot(statistics["train_loss"])
        # axs[0].set_title("Train loss")
        # axs[1].plot(statistics["train_accuracy"])
        # axs[1].set_title("Train accuracy")
        # fig.savefig("reports/figures/training_statistics.png")


if __name__ == '__main__':
    train()