import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
import os
import wandb
from tqdm import tqdm
import argparse
from Data.CelebA import CustomCelebA

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attr_indices = {
            'BlondHair': 9,     # Target attribute for hair color
            'HeavyMakeup': 18,  # Alternative target attribute
        }
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        # Define transformations
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create datasets
        self.train_dataset = CustomCelebA(
            root=self.config.data_dir,
            split='train',
            target_type="attr",
            transform=train_transform,
        )
        
        self.val_dataset = CustomCelebA(
            root=self.config.data_dir,
            split='valid',
            target_type="attr",
            transform=val_transform,
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def setup_model(self):
        self.model = models.resnet18(num_classes=2, weights=None).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # TODO: inconsisten use of self.config.key and self.config['key']
        if self.config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'])
            
        elif self.config['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'])
        
        if self.config['lr_scheduler']:
            # learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.1, 
                patience=5
            )
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets[:, self.attr_indices[self.config['target_attr']]].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
            
            # Log metrics to wandb
            wandb.log({
                "train_loss": loss.item(),
                "train_acc": 100.*correct/total,
                "epoch": epoch
            })
        
        return running_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                targets = targets[:, self.attr_indices[self.config['target_attr']]].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss/len(self.val_loader)
        val_acc = 100.*correct/total
        
        # Log validation metrics to wandb
        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch
        })
        
        # Update learning rate scheduler
        if self.config.lr_scheduler:
            self.scheduler.step(val_acc)
        
        return val_loss, val_acc
    
    def train(self):
        best_acc = 0
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"Saving best model with accuracy: {best_acc:.2f}%")
                
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': best_acc
                }
                
                torch.save(checkpoint, os.path.join(
                    self.config.output_dir, 
                    f"resnet18_{self.config.target_attr}_best.pth"
                ))
                
                # Log best model to wandb
                wandb.run.summary["best_accuracy"] = best_acc
                wandb.run.summary["best_epoch"] = epoch


def main():
    parser = argparse.ArgumentParser(description="Wandb setting")
    parser.add_argument("--sweep", action='store_true', help="Perform a parameter sweep")
    args = parser.parse_args()
    
    if not args.sweep:
        config = dict(
            data_dir = 'PATH\\TO\\DATA_DIR',
            output_dir = 'PATH\\TO\\OUTPUT_DIR',
            num_workers = 2,
            target_attr = 'BlondHair',  # 'BlondHair' | 'HeavyMakeup'
            batch_size = 256,
            optimizer = "Adam",
            learning_rate = 0.001,
            weight_decay = 1e-4,
            lr_scheduler = False,
            epochs = 5,
        )
        
        wandb.login(key='INSERISCI WANDB_KEY')
        with wandb.init(entity="learning-from-failure", project="resnet", id="debugRun", config=config, mode="offline"):
        # with wandb.init(entity="learning-from-failure", project="resnet", id="debugRun", config=config): #vars(args)
            trainer = Trainer(wandb.config)
            trainer.train()
            wandb.finish()
    else:
        def sweep_train(config=None):
            with wandb.init(config=config):
                config = wandb.config
                trainer = Trainer(config)
                trainer.train()

        # For sweep
        sweep_config = dict(
                method="random",
                metric=dict(name="val/accuracy", goal="maximize"),
                parameters=dict(
                    learning_rate=dict(min=1e-4, max=1e-3),
                    batch_size=dict(values=[128, 256]),
                    weight_decay=dict(values=[1e-4, 1e-3]),
                    optimizer=dict(values=["Adam", "AdamW"]),
                    target_attr=dict(value="BlondHair"),
                    epochs=dict(value=5),
                    data_dir=dict(value="PATH/TO/DATA_DIR"),
                ),
            )

        sweep_id = wandb.sweep(sweep_config, project="resnet18-celeba-sweep")
        wandb.agent(sweep_id, sweep_train, count=10)

if __name__ == "__main__":
    # For regular training
    main()