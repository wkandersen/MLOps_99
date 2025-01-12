# Validation
import torch
from train import train
from model import CustomResNet50, SimpleCNN
from data import load_data
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import resnet50
writer = SummaryWriter() # For Tensorboard
import hydra

model_name = 'models/model.pth'
cuda = False

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")

def evaluate(config):
    model_checkpoint = 'models/model.pth'
    writer = SummaryWriter() # For Tensorboard
    early_stop_count=0
    ES_patience=5
    best = 0.0

    print("Evaluating like my life depended on it")
    hparams = config['hyperparameters']
    model = SimpleCNN(num_classes=hparams['num_classes'], x_dim = hparams['x_dim'])
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    #Initialize 
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), hparams["lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1,patience=1,verbose=True)

    for epoch in range(hparams['epochs']):
        _, valid_loader, _ = load_data()
        with torch.no_grad():
            correct = 0
            val_loss = 0.0
            vbar = tqdm(valid_loader, desc = 'Validation', position=0, leave=True)
            for i,(inp,lbl) in enumerate(vbar):
                if cuda:
                    inp,lbl = inp.cuda(),lbl.cuda()
                out = model(inp)
                val_loss += criterion(out,lbl)
                out = out.argmax(dim=1)
                correct += (out == lbl).sum().item()
            val_acc = 100.0*correct/len(valid_loader.dataset)
            val_loss /= (len(valid_loader.dataset)/hparams['batch_size'])
        print(f'\nEpoch: {epoch+1}/{hparams["epochs"]}')
        print(f'Validation loss: {val_loss}, Validation Accuracy: {val_acc}\n')

        scheduler.step(val_loss)

        # write to tensorboard
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_acc>best:
            best=val_acc
            torch.save(model,model_name)
            early_stop_count=0
            print('Accuracy Improved, model saved.\n')
        else:
            early_stop_count+=1

        if early_stop_count==ES_patience:
            print('Early Stopping Initiated...')
            print(f'Best Accuracy achieved: {best:.2f}% at epoch:{epoch-ES_patience}')
            print(f'Model saved as {model_name}')
            break

    
    writer.flush()

if __name__ == "__main__":
    evaluate()