import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader,  val_loader,   config , device=None,):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = config.lr
        self.epochs = config.epoch
        self.save_path = config.save_path

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        
    def train(self):
        best_val_acc = 0.0
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                  f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # 保存最优模型
            if self.save_path and val_acc > best_val_acc:
                torch.save(self.model.state_dict(), self.save_path)
                best_val_acc = val_acc
                print(f"模型已保存到: {self.save_path}")

    def validate(self):
        if self.val_loader is None:
            return 0.0, 0.0
        self.model.eval()
        total, correct = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return val_loss / len(self.val_loader), 100. * correct / total



    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        total, correct = 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{running_loss/len(self.train_loader):.4f}",
                             acc=f"{100.*correct/total:.2f}%")

        return running_loss / len(self.train_loader), 100. * correct / total
