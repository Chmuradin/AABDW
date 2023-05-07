import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float,float]:
    """
    Trains PyTorch model for a single epoch
      Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy)
    """
    #Putting model in training mode
    model.train()

    #Setup train loss and train accuracy values
    train_loss, train_acc=0,0

    #Loop through data loader data batches
    for batch, (X,y) in enumerate(dataloader):
        #Send the data to the target device
        X,y=X.to(device), y.to(device)
        #Forward pass
        y_pred=model(X)
        #Calculate and accumulate the loss
        loss=loss_fn(y_pred,y)
        train_loss+=loss.item()
        #Optimizer zero grad
        optimizer.zero_grad()
        #Loss backward
        loss.backward()
        #Optimizer step
        optimizer.step()
        #Calculate and accumulate accuracy metric across all batches
        y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc+=(y_pred_class==y).sum().item()/len(y_pred)
    #Adjust metrics to get average loss and accuracy per batch
    train_loss=train_loss/len(dataloader)
    train_acc=train_acc/len(dataloader)
    return train_loss,train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.
  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy)
  """
  #putting in evaluation mode
  model.eval()
  #setup test loss and test accuracy values
  test_loss, test_acc=0,0
  #Turn on inference context manager
  with torch.inference_mode():
     #Looping through dataloader batches
     for batch, (X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)
        test_pred_logits=model(X)
        loss=loss_fn(test_pred_logits,y)
        test_loss+=loss.item()
        test_pred_labels=test_pred_logits.argmax(dim=1)
        test_Acc+=((test_pred_labels==y).sum().item()/len(test_pred_labels))
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
   """
   Trains and tests PyTorch model.
    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.
    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    """
   #create dictionary
   results={'train_loss':[],
            'test_loss':[],
            'test_acc':[]}
   #Loop through training and testing steps for a number of epochs
   for epoch in tqdm(range(epochs)):
      train_loss,train_acc,=train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
      test_loss,test_acc=test_step(model=model,
                                   dataloader=test_dataloader,
                                   loss_fn=loss_fn,
                                   device=device)
      #printing the progress
      print(f'Epoch: {epoch+1} | '
            f'train_loss: {train_loss} | '
            f'train_acc: {train_acc} | '
            f'test_loss: {test_loss} |'
            f'test_acc: {test_acc}')
      
      #updating results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
    #returning the filled results at the end of the epochs
   return results