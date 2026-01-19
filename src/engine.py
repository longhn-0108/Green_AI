import torch

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Huấn luyện 1 epoch."""
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    """Đánh giá mô hình."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = correct / total
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, acc