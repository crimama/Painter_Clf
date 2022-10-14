import torch 
from sklearn.metrics import f1_score 
import numpy as np 
def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

def epoch_run(data_loader,model,device,optimizer,criterion,scheduler,scaler):
    model.train()
    epoch_loss = 0 
    epoch_pred = [] 
    epoch_target = [] 
    i = 0 
    for batch_x,batch_labels in data_loader:
        batch_x = batch_x.to(device)
        batch_labels = batch_labels.to(device)
        with torch.cuda.amp.autocast():
            pred = model(batch_x)   
        loss = criterion(pred,batch_labels)
        
        #Backpropagation 
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()/len(data_loader)
        epoch_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        epoch_target += batch_labels.detach().cpu().numpy().tolist()
    scheduler.step()
    train_f1 = score_function(epoch_pred,epoch_target)
    
    return epoch_loss, train_f1 

def valid_epoch_run(data_loader,model,device,criterion):
    model.eval()
    epoch_loss = []
    epoch_pred = [] 
    epoch_target = [] 
    i = 0 
    with torch.no_grad():
        for batch_x,batch_labels in data_loader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)
            with torch.cuda.amp.autocast():
                pred = model(batch_x)   
            loss = criterion(pred,batch_labels)

            epoch_loss.append(loss.item())
            epoch_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            epoch_target += batch_labels.detach().cpu().numpy().tolist()
    
    train_f1 = score_function(epoch_pred,epoch_target)
    
    return np.mean(epoch_loss), train_f1 