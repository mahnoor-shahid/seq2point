import os.path
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.training_utilities import early_stopping, set_GPU
import datetime
import numpy as np
from utils.compute_metrics import compute_TP, compute_recall, compute_precision, compute_f1


def fetch_training_reports(epoch, batch_idx, timestep, y_value, prediction):

    if (compute_TP(y_value, prediction, TRAINING_CONFIG['THRESHOLD'])/TRAINING_CONFIG['TRAIN_BATCH_SIZE']) > 0.3:
        tmp_df = pd.DataFrame({'time': timestep, 'ground_truth': y_value, 'prediction': prediction})
        training_path = os.path.join(TRAINING_CONFIG['EXPERIMENT_PATH'], 'training')
        tmp_df.to_csv(f'{training_path}/epoch_{epoch}_batch_{batch_idx}.csv')

def network_train(model, criterion, optimizer, train_loader, validation_loader, assess_training=False):
    """
    """
    try:
        print("\n\nTraining the model architecture...")

        writer = SummaryWriter()
        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        recall_per_epoch = []
        precision_per_epoch = []
        total_training_time = []

        best_loss, idle_training_epochs = None, 0
        # writer = SummaryWriter(comment='_training_visualization')

        if not os.path.exists(os.path.join(TRAINING_CONFIG['EXPERIMENT_PATH'], 'training')):
            os.makedirs(os.path.join(TRAINING_CONFIG['EXPERIMENT_PATH'], 'training'))

        for epoch in range(0, TRAINING_CONFIG['NUM_EPOCHS']):

            if early_stopping(idle_training_epochs):
                break

            train_loss_scores = []
            validation_loss_scores = []
            recall_scores = []
            precision_scores = []

            start_training_time = datetime.datetime.now()
            model.train()
            for batch_idx, (timestep, x_value, y_value) in enumerate(train_loader):

                x_value = x_value[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                y_value = y_value[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                prediction = model.forward(x_value)[:, None].type(torch.cuda.FloatTensor).to(set_GPU())

                loss = criterion(y_value, prediction)
                train_loss_scores.append(loss.item())

                timestep = [datetime.datetime.fromtimestamp(each_timestep).strftime('%Y-%m-%d %H:%M:%S') for
                            each_timestep in
                            timestep.numpy()]
                y_value, prediction = y_value.cpu().detach().numpy().flatten(), prediction.cpu().detach().numpy().flatten()

                recall_scores.append(compute_recall(y_value, prediction, threshold=10.0))
                precision_scores.append(compute_precision(y_value, prediction, threshold=10.0))

                if assess_training:
                    fetch_training_reports(epoch, batch_idx, timestep, y_value, prediction)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx+1) % 50 == 0:
                    print(f"Epoch : [{epoch+1}/{TRAINING_CONFIG['NUM_EPOCHS']}] | Step : [{batch_idx+1}/{len(train_loader)}]|  Average Training Loss : {np.mean(train_loss_scores)}")

            training_loss_per_epoch.append(np.mean(train_loss_scores))
            recall_per_epoch.append(np.mean(recall_scores))
            precision_per_epoch.append(np.mean(precision_scores))

            writer.add_scalar("training_loss_per_epoch", training_loss_per_epoch[-1], epoch)
            writer.add_scalar("recall_per_epoch", recall_per_epoch[-1], epoch)
            writer.add_scalar("precision_per_epoch", precision_per_epoch[-1], epoch)

            model.eval()
            with torch.no_grad():
                for batch_idx, (timestep, x_value, y_value) in enumerate(validation_loader):
                    x_value = x_value[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                    y_value = y_value[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                    prediction = model.forward(x_value)[:, None].type(torch.cuda.FloatTensor).to(set_GPU())

                    loss = criterion(y_value, prediction)
                    validation_loss_scores.append(loss.item())

                    if (batch_idx+1) % 20 == 0:
                        print(f"Epoch : [{epoch+1}/{TRAINING_CONFIG['NUM_EPOCHS']}] | Step : [{batch_idx+1}/{len(validation_loader)}]|  Average Validation Loss : {np.mean(validation_loss_scores)}")

            validation_loss_per_epoch.append(np.mean(validation_loss_scores))
            writer.add_scalar("validation_loss_per_epoch", validation_loss_per_epoch[-1], epoch)

            end_training_time = datetime.datetime.now()
            total_training_time.append(end_training_time-start_training_time)
            print("==================================================================================================================================================")
            print(f"Epoch : [{epoch+1}/{TRAINING_CONFIG['NUM_EPOCHS']}] | Training Loss : {training_loss_per_epoch[-1]}, | Validation Loss : {validation_loss_per_epoch[-1]}, | Time consumption: {total_training_time[-1]}s")
            print("==================================================================================================================================================")

            checkpoint_loss = validation_loss_per_epoch[-1]

            if best_loss is None:
                best_loss = checkpoint_loss
                idle_training_epochs = idle_training_epochs + 1
            elif best_loss <= checkpoint_loss:
                idle_training_epochs = idle_training_epochs + 1
            elif best_loss > checkpoint_loss:
                best_loss = checkpoint_loss
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
        print("******************************************************************************************************************************************************")
        print( f"Total Training Time : {np.array(total_training_time).sum()}s")
        print("******************************************************************************************************************************************************")
        writer.flush()
        writer.close()
        return (training_loss_per_epoch, validation_loss_per_epoch, recall_per_epoch, precision_per_epoch)
    
    except Exception as e:
        print("Error occured in network_train method due to ", e)