
import torch
# from torch.utils.tensorboard import SummaryWriter ### in progress
from utils.training_utilities import early_stopping, set_GPU
import datetime
import numpy as np


def network_training(model, criterion, optimizer, train_loader, validation_loader):
    """
    """
    try:
        print(model)
        print("\nTraining the model architecture...")     
        training_loss_per_epoch = []
        validation_loss_per_epoch = []

        best_loss, idle_training_epochs = None, 0
        # writer = SummaryWriter(comment='_training_visualization')

        for epoch in range(0, TRAINING_CONFIG['NUM_EPOCHS']):

            if early_stopping(idle_training_epochs):
                break
            
            start_training_time = datetime.datetime.now()
            model.train()
            train_loss_scores = []
            validation_loss_scores= []

            for batch_idx, (data, target) in enumerate(train_loader):

                data = data[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                target = target[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                predictions = model.forward(data)[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                loss = criterion(target, predictions)
                train_loss_scores.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx+1) % 20 == 0:
                    print(f"Epoch : [{epoch+1}/{TRAINING_CONFIG['NUM_EPOCHS']}] | Step : [{batch_idx+1}/{len(train_loader)}]|  Average Training Loss : {np.mean(train_loss_scores)}")
    
            training_loss_per_epoch.append(np.mean(train_loss_scores))
            
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(validation_loader):                                              
                    data = data[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                    target = target[:, None].type(torch.cuda.FloatTensor).to(set_GPU())                    
                    predictions = model.forward(data)[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                    
                    loss = criterion(target, predictions)
                    validation_loss_scores.append(loss.item())
                    
                    if (batch_idx+1) % 20 == 0:
                        print(f"Epoch : [{epoch+1}/{TRAINING_CONFIG['NUM_EPOCHS']}] | Step : [{batch_idx+1}/{len(validation_loader)}]|  Average Validation Loss : {np.mean(validation_loss_scores)}")

                    
            validation_loss_per_epoch.append(np.mean(validation_loss_scores))
            
            end_training_time = datetime.datetime.now()
            print("==================================================================================================================================================")
            print(f"Epoch : [{epoch}/{TRAINING_CONFIG['NUM_EPOCHS']}] | Training Loss : {training_loss_per_epoch[-1]}, | Validation Loss : {validation_loss_per_epoch[-1]}, | Time consumption: {end_training_time-start_training_time}s")
            
            if best_loss is None or best_loss <= validation_loss_per_epoch[-1]:
                best_loss = validation_loss_per_epoch[-1]
                idle_training_epochs = idle_training_epochs + 1
            elif best_loss > validation_loss_per_epoch[-1]:
                best_loss = validation_loss_per_epoch[-1]
                idle_training_epochs = 0
                time = datetime.datetime.now().date()
                model.save_model(filename=f"{time}_best_loss_{round(best_loss)}")  
            else:
                pass
            
            for name,param in model.named_parameters():
                pass
                # writer.add_histogram(name + '_grad', param.grad, epoch)
                # writer.add_histogram(name + '_data', param, epoch)
            # writer.add_scalars("Bleh", {"Check":best_loss}, epoch)
        
        return training_loss_per_epoch, validation_loss_per_epoch
    
    except Exception as e:
        print("Error occured in network_training method due to ", e)