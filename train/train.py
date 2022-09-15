
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

def network_training():
    """
    """
    try:
        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        best_loss, idle_training_epochs = None, 0
        writer = SummaryWriter(comment='training_visuaization')

        for epoch in range(0, training_config['NUM_EPOCHS']):

            if early_stopping(idle_training_epochs):
                break

            model.train()
            start_training_time = time.time() 
            train_loss_scores = []
            validation_loss_scores= []

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(set_GPU())

                # predictions = model.forward(data.to(set_GPU()))
                predictions = torch.randn(1,1)

                loss = criterion(predictions.to(set_GPU()), target.to(set_GPU()))
                train_loss_scores.append(loss.item())
                
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # if (batch_idx+1) % 100 == 0:
                #     print(f"Epoch : [{epoch+1}/{training_config['NUM_EPOCHS']}] | Step : [{batch_index+1}/{len(train_loader)}] | Loss : {loss.item()}")
            
            training_loss_per_epoch.append(np.min(train_loss_scores))
            
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(validation_loader):
                    data = data.to(set_GPU())
                    
                    # predictions = model.forward(data.to(set_GPU()))
                    predictions = torch.randn(1,1)
                    
                    loss = criterion(predictions.to(set_GPU()), target.to(set_GPU()))
                    validation_loss_scores.append(loss.item())
                    
            validation_loss_per_epoch.append(np.min(validation_loss_scores))
            
            end_training_time = time.time()
            print(f"Epoch : [{epoch}/{training_config['NUM_EPOCHS']}] | Training Loss : {training_loss_per_epoch[-1]}, | Validation Loss : {validation_loss_per_epoch[-1]}, | Time consumption: {end_training_time-start_training_time}s")
            
            if best_loss is None:
                best_loss = validation_loss_per_epoch[-1]
            elif best_loss > validation_loss_per_epoch[-1]:
                best_loss = validation_loss_per_epoch[-1]
                idle_training_epochs = 0
                model.save_model(filename=f"{best_loss}")  
            else:
                idle_training_epochs +=1
            
            for name,param in model.named_parameters():
                pass
                # writer.add_histogram(name + '_grad', param.grad, epoch)
                # writer.add_histogram(name + '_data', param, epoch)
            writer.add_scalars("Bleh", {"Check":best_loss}, epoch)
        
        return training_loss_per_epoch, validation_loss_per_epoch
    
    except Exception as e:
        print("Error occured in network_training method due to ", e)