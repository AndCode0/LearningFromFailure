import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import argparse

from SimpleConv import SimpleConvNet

class VanillaTrainer:
    def __init__(self, config, dataset, model):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.model = model.to(self.device)
        
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        self.train_loader = DataLoader(
            self.dataset['train'],
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.dataset['valud'],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def setup_model(self):
        self.criterion = nn.CrossEntropyLoss()
        
        if self.config.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        if self.config.lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.1, patience=5
            )
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/(total+1), 'acc': 100.*correct/total})
            
            wandb.log({"train_loss": loss.item(), "train_acc": 100.*correct/total, "epoch": epoch})
        
        return running_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self, epoch):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss, val_acc = running_loss/len(self.val_loader), 100.*correct/total
        wandb.log({"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})
        
        if self.config.lr_scheduler:
            self.scheduler.step(val_acc)
        
        return val_loss, val_acc
    
    def train(self):
        best_acc = 0
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\nVal Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"Saving best model with accuracy: {best_acc:.2f}%")
                torch.save(self.model.state_dict(), f"{self.config.output_dir}/best_model.pth")
                wandb.run.summary["best_accuracy"] = best_acc
                wandb.run.summary["best_epoch"] = epoch


def main():
    parser = argparse.ArgumentParser(description="Training Configuration")
    args = parser.parse_args()
    
    config = {
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "Adam",
        "lr_scheduler": False,
        "epochs": 20,
        "num_workers": 1,
        "output_dir": "./outputs"
    }
    
    # Initialize dataset and model
    dataset = {"train": ..., "val": ...}  # Load dataset here
    model = SimpleConvNet(num_classes=10)
    
    wandb.login(key='YOUR_WANDB_KEY')
    with wandb.init(entity="learning-from-failure", project="SimpleConv_nodecay", id="run_nodecay_20ep1", config=config):
        trainer = VanillaTrainer(wandb.config, dataset, model)
        trainer.train()
        wandb.finish()

if __name__ == "__main__":
    main()
