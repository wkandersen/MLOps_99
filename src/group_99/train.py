from model import CustomResNet50
from data import load_data
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import hydra

# loading
cuda = False

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.1")

def train(config):
    hparams = config['hyperparameters']

    for epoch in range(hparams['epochs']):
        # Training

        # Model, optimizer, criterion initialization
        model = CustomResNet50(num_classes=hparams['num_classes'], pretrained=True)
        optimizer = Adam(model.parameters(), hparams["lr"])
        criterion = CrossEntropyLoss()

        # Load data
        train_loader, valid_loader = load_data()

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

if __name__ == '__main__':
    train()