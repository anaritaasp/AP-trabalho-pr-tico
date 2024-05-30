import logging
import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, ScaleIntensity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def evaluate_new_images(epochs):
    """
    Avaliar novas imagens usando um modelo treinado.

    Args:
        model_path (str): Caminho para o modelo treinado.
        new_images_path (str): Caminho para as imagens sintéticas.

    Returns:
        tuple: Listas de boas imagens, imagens más e muito más, com base nas probabilidades.
    """
    print("A carregar o modelo treinado...")
    model_path = f"classification_model_{epochs}.pth"
    new_images_path = os.sep.join(["synthetic_image_test"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])

    print("A carregar imagens sintéticas...")
    new_images = [os.path.join(new_images_path, file) for file in os.listdir(new_images_path) if file.endswith(".nii.gz")]
    new_ds = ImageDataset(image_files=new_images, labels=None, transform=transforms)
    new_loader = DataLoader(new_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

    all_probs = []

    print("A avaliar imagens sintéticas...")
    with torch.no_grad():
        for batch_data in new_loader:
            inputs = batch_data.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)

    # Categorizar as imagens com base nas probabilidades
    good_images = [img.split('\\')[-1] for img, prob in zip(new_images, all_probs) if prob >= 0.5]
    bad_images = [img.split('\\')[-1] for img, prob in zip(new_images, all_probs) if 0.1 <= prob < 0.5]
    very_bad_images = [img.split('\\')[-1] for img, prob in zip(new_images, all_probs) if prob < 0.2]

    with open(f"{epochs} epochs/results{epochs}.txt", "w") as f:
        f.write("Good Images:\n")
        for img in good_images:
            f.write(img + "\n")
        f.write("\nBad Images:\n")
        for img in bad_images:
            f.write(img + "\n")
        f.write("\nVery Bad Images:\n")
        for img in very_bad_images:
            f.write(img + "\n")
        
    print("Avaliação concluída.")
    

def plot_data_balance(labels_train, labels_val, filename_suffix):
    """
    Plotar a distribuição de dados de treino e validação.

    Args:
        labels_train (list): Lablels do conjunto de treino.
        labels_val (list): Labels do conjunto de validação.
        filename_suffix (str): Sufixo para o nome do ficheiro do gráfico.
    """
    print("A gerar gráfico da distribuição dos dados de treino e validação...")
    train_fake_count = sum(1 for label in labels_train if label == 0)
    train_real_count = sum(1 for label in labels_train if label == 1)

    val_fake_count = sum(1 for label in labels_val if label == 0)
    val_real_count = sum(1 for label in labels_val if label == 1)

    labels = ['Fake', 'Real']
    train_counts = [train_fake_count, train_real_count]
    val_counts = [val_fake_count, val_real_count]

    bar_width = 0.35
    x_train = np.arange(len(labels))
    x_val = [x + bar_width for x in x_train]

    plt.bar(x_train, train_counts, width=bar_width, label='Train', color='blue', alpha=0.5)
    plt.bar(x_val, val_counts, width=bar_width, label='Validation', color='orange', alpha=0.5)

    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('Counts of Fake and Real labels in Train and Validation sets')
    plt.xticks([x + bar_width / 2 for x in x_train], labels)
    plt.legend()

    plt.savefig(f"{epochs} epochs/{filename_suffix}_data_balance.png") 
    plt.close()  
    print("Gráfico da distribuição dos dados gerado e guardado.")

def plot_labels_distribution(labels_train, epochs, filename_suffix):
    """
    Plotar a distribuição da labels no conjunto de treino.

    Args:
        labels_train (list): labels do conjunto de treino.
        epochs (int): Número de épocas.
        filename_suffix (str): Sufixo para o nome do ficheiro do gráfico.
    """
    print("A gerar gráfico da distribuição das labels no conjunto de treino...")
    class_names = ['fake', 'real']
    label_names = [class_names[label] for label in labels_train]

    plt.figure(figsize=(10, 5))
    plt.plot(label_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.title('Labels in Training Set')
    plt.xlabel('Image Index')
    plt.ylabel('Label')
    plt.savefig(f"{epochs} epochs/{epochs}_labels_distribution_{filename_suffix}.png") 
    plt.close()  
    print("Gráfico da distribuição das labels gerado e guardado.")

def plot_accuracy(epochs_ranges, train_accuracies, val_accuracies):
    """
    Plotar a acurácia do treino e validação ao longo das épocas.

    Args:
        epochs (list): Lista de épocas.
        train_accuracies (list): Lista de precisão do treino.
        val_accuracies (list): Lista de precisão da validação.
    """
    print("A gerar gráfico da accuracy do treino e validação...")
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_ranges, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs_ranges, val_accuracies, 'orange', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{len(epochs_ranges)} epochs/{len(epochs_ranges)}_accuracy_over_epochs.png")  
    plt.close() 
    print("Gráfico da accuracy gerado e guardado.")

def plot_loss(epochs_ranges, train_loss_values, val_loss_values):
    """
    Plotar a perda do treino e validação ao longo das épocas.

    Args:
        epochs (list): Lista de épocas.
        train_loss_values (list): Lista de valores de perda do treino.
        val_loss_values (list): Lista de valores de perda da validação.
    """
    print("A gerar gráfico da perda do treino e validação...")
    plt.figure()
    plt.plot(epochs_ranges, train_loss_values, 'b-', label='Training loss')
    plt.plot(epochs_ranges, val_loss_values, 'orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{len(epochs_ranges)} epochs/{len(epochs_ranges)}_training_and_validation_loss.png")
    plt.close()
    print("Gráfico da perda gerado e guardado.")

def prepare_data():
    """
    Preparar dados de treino e validação.

    Returns:
        tuple: Listas de labels corretas e imagens.
    """
    print("A preparar dados de treino e validação...")
    fake_data_path = os.sep.join(["original_data", "fake"])
    real_data_path = os.sep.join(["original_data", "real"])

    fake_images = [os.path.join(fake_data_path, file) for file in os.listdir(fake_data_path) if file.endswith(".nii.gz")]
    real_images = [os.path.join(real_data_path, file) for file in os.listdir(real_data_path) if file.endswith(".nii.gz")]

    correct_labels = [0] * len(fake_images) + [1] * len(real_images)
    images = fake_images + real_images

    print("Dados preparados.")
    return correct_labels, images

def prepare_transforms():
    """
    Preparar transformações para os dados de treino e validação.

    Returns:
        tuple: Transformações de treino e validação.
    """
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])
    return train_transforms, val_transforms

def prepare_datasets(correct_labels, images, train_transforms, val_transforms, epochs):
    """
    Preparar conjuntos de dados de treino e validação.

    Args:
        correct_labels (list): Lista de labels (labels sintéticas e reais).
        images (list): Lista de imagens (sintéticas e reais).
        train_transforms (Compose): Transformações de treino.
        val_transforms (Compose): Transformações de validação.
        epochs (int): Número de épocas.

    Returns:
        tuple: Conjunto de dados de treino, loader de treino, loader de validação e índices de validação.
    """
    print("A preparar conjuntos de dados de treino e validação...")
    train_indices = list(range(35)) + list(range(50, 85))
    val_indices = list(range(35, 50)) + list(range(85, 100))

    image_files_train = [images[i] for i in train_indices]
    labels_train = [correct_labels[i] for i in train_indices]
    images_files_val = [images[i] for i in val_indices]
    labels_val = [correct_labels[i] for i in val_indices]

    plot_data_balance(labels_train, labels_val, epochs)

    train_ds = ImageDataset(image_files=image_files_train, labels=labels_train, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    val_ds = ImageDataset(image_files=images_files_val, labels=labels_val, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

    print("Conjuntos de dados preparados.")
    return train_ds, train_loader, val_loader, val_indices

def train(model, optimizer, train_loader, loss_function, device, epochs, epoch, plot_counter, train_ds):
    """
    Treinar o modelo.

    Args:
        model (torch.nn.Module): O modelo a ser treinado.
        optimizer (torch.optim.Optimizer): Otimizador.
        train_loader (DataLoader): Loader de treino.
        loss_function (torch.nn.Module): Função de perda.
        device (torch.device): Dispositivo (CPU ou GPU).
        epochs (int): Número de épocas.
        epoch (int): Época atual.
        plot_counter (int): Contador para os gráficos.
        train_ds (Dataset): Conjunto de dados de treino.

    Returns:
        tuple: Perda e accuracy de treino.
    """
    print(f"Treino - Epoch {epoch + 1}")
    model.train()
    epoch_loss = 0
    step = 0
    probs_train = list()
    labels_train = list()

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}")

        val_probs = torch.sigmoid(outputs)
        probs_train.extend(val_probs.squeeze().detach().cpu().numpy())
        labels_train.extend(labels.squeeze().detach().cpu().numpy())

    rounded_outputs = np.round(probs_train)
    correct = np.sum(np.equal(rounded_outputs, labels_train))

    metric = correct / len(labels_train)
    
    plot_labels_distribution(labels_train, epochs, f"epoch_{epoch + 1}_plot_{plot_counter}")
    plot_counter += 1

    epoch_loss /= step
    
    return epoch_loss, metric

def validate(model, val_loader, loss_function, device):
    """
    Validar o modelo.

    Args:
        model (torch.nn.Module): O modelo a ser validado.
        val_loader (DataLoader): Loader de validação.
        loss_function (torch.nn.Module): Função de perda.
        device (torch.device): Dispositivo (CPU ou GPU).

    Returns:
        tuple: Perda e accuracy de validação e probabilidades.
    """
    print("Validação do modelo...")
    model.eval()
    val_loss = 0

    with torch.no_grad():
        probs_squeeze = []
        labels_squeeze = []
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images)
            val_probs = torch.sigmoid(val_outputs)
            val_loss += loss_function(val_outputs.squeeze(), val_labels.float()).item()

            probs_squeeze.extend(val_probs.squeeze().cpu().numpy())
            labels_squeeze.extend(val_labels.squeeze().cpu().numpy())

        rounded_outputs = np.round(probs_squeeze)
        correct = np.sum(np.equal(rounded_outputs, labels_squeeze))

        val_loss /= len(val_loader)
        metric = correct / len(labels_squeeze)

    print("Validação concluída.")
    return val_loss, metric, probs_squeeze

def main(epochs):
    """
    Função principal que executa o treino e validação do modelo.
    """
    model_name = "DenseNet121"
    plot_counter = 1  

    folder_name = f"{epochs} epochs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open(f"{folder_name}/{epochs}_{model_name}.txt", "w") as f:
        f.write(model_name)
        f.write("\n")
        monai.config.print_config()
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    correct_labels, images = prepare_data()
    train_transforms, val_transforms = prepare_transforms()

    train_ds, train_loader, val_loader, val_indices = prepare_datasets(correct_labels, images, train_transforms, val_transforms, epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    best_metric = -1
    epoch_loss_values = list()
    val_loss_values = list()
    metric_values = list()
    metric_train = list()

    for epoch in range(epochs):
        print("-" * 10)
        print(f"Época {epoch + 1}/{epochs}")

        with open(f"{epochs} epochs/{epochs}_{model_name}.txt", "a") as f:
            f.write("-" * 10)
            f.write(f"Época {epoch + 1}/{epochs}\n")

        epoch_loss, metric_train_epoch = train(model, optimizer, train_loader, loss_function, device, epochs, epoch, plot_counter, train_ds)
        epoch_loss_values.append(epoch_loss)
        metric_train.append(metric_train_epoch)

        val_loss, metric_val, probs_squeeze = validate(model, val_loader, loss_function, device)
        val_loss_values.append(val_loss)
        metric_values.append(metric_val)

        if metric_val > best_metric:
            best_metric = metric_val
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"classification_model_{epochs}.pth")

        with open(f"{epochs} epochs/{epochs}_{model_name}.txt", "a") as f:
            for index, prob in zip(val_indices, probs_squeeze):
                filename = images[index].split('\\')[-1]
                labels = correct_labels[index]
                f.write(f"Filename: {filename}, Label: {labels}, Probability: {prob}\n")
            f.write(f"Época atual: {epoch + 1}; Acurácia de validação atual: {metric_val:.4f}; Melhor acurácia de validação: {best_metric:.4f} na época {best_metric_epoch};\n")

        plot_counter += 1

    epochs_range = range(1, epochs + 1)

    plot_accuracy(epochs_range, metric_train, metric_values)
    plot_loss(epochs_range, epoch_loss_values, val_loss_values)

    print(f"Treino concluído, melhor métrica: {best_metric:.4f} na época: {best_metric_epoch}")

if __name__ == "__main__":
    epochs = 5
    if os.path.exists(f"classification_model_{epochs}.pth"):
        evaluate_new_images(epochs)
    else:
        main(epochs)
        evaluate_new_images(epochs)
