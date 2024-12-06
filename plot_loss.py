import matplotlib.pyplot as plt
import os

def parse_loss_file(loss_file_path):
    """
    Parse the loss.txt file to extract training and validation loss for each epoch.

    :param loss_file_path: Path to the loss.txt file
    :return: Tuple of lists (train_losses, val_losses)
    """
    train_losses = []
    val_losses = []

    with open(loss_file_path, 'r') as f:
        for line in f:
            if "Epoch" in line:  # Ensure we're processing lines that contain epoch info
                parts = line.strip().split(',')
                train_loss = float(parts[0].split('=')[1].strip())
                val_loss = float(parts[1].split('=')[1].strip())
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
    return train_losses, val_losses

def plot_losses(model_name, plot_title, base_local_path='models', save_path=None):
    """
    Plot the training and validation loss across epochs for the given model and save the plot to a file.

    :param model_name: The model's name whose losses are being plotted
    :param base_local_path: The base local path where the model data is stored
    :param save_path: The file path to save the plot (if None, the plot will not be saved)
    """
    epochs = range(1, 16)  # Assuming e1 to e15 correspond to Epochs 1 to 15
    all_train_losses = []
    all_val_losses = []

    for epoch in epochs:
        # Local path to the loss.txt file for the current epoch
        loss_file_path = os.path.join(base_local_path, model_name, f'e{epoch}', 'loss.txt')
        
        try:
            # Parse the loss.txt file to get the losses
            train_losses, val_losses = parse_loss_file(loss_file_path)
            
            # Only expect 1 loss entry for each epoch, so take the first one
            all_train_losses.append(train_losses[0])
            all_val_losses.append(val_losses[0])
        except Exception as e:
            print(f"Failed to parse loss file for epoch {epoch}: {e}")
            continue

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), all_train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, 16), all_val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {plot_title}')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

model_name = "fine_tuned_wav2vec_t5_frozen"
plot_title = "Frozen wav2vec 2.0 to T5"
save_path = "plots/" + model_name + '_loss_plot.png' 
plot_losses(model_name, plot_title, save_path=save_path)

model_name = "fine_tuned_wav2vec_t5_unfrozen" 
plot_title = "Unfrozen wav2vec 2.0 to T5"
save_path = "plots/" + model_name + '_loss_plot.png'
plot_losses(model_name, plot_title, save_path=save_path)

model_name = "fine_tuned_mert_pro_t5_frozen" 
plot_title = "Frozen Mert to T5"
save_path = "plots/" + model_name + '_loss_plot.png'
plot_losses(model_name, plot_title, save_path=save_path)

model_name = "fine_tuned_mert_pro_t5_unfrozen" 
plot_title = "Unfrozen Mert to T5"
save_path = "plots/" + model_name + '_loss_plot.png'
plot_losses(model_name, plot_title, save_path=save_path)

model_name = "fine_tuned_clap_t5_frozen" 
plot_title = "Frozen Clap to T5"
save_path = "plots/" + model_name + '_loss_plot.png'
plot_losses(model_name, plot_title, save_path=save_path)

model_name = "fine_tuned_clap_t5_unfrozen" 
plot_title = "Unfrozen Clap to T5"
save_path = "plots/" + model_name + '_loss_plot.png'
plot_losses(model_name, plot_title, save_path=save_path)