import logging
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, ScaleIntensity

import torch.nn.functional as F

def main():
    model_name = "ResNet"
    #model_name = "DenseNet121"
    # Abrir um ficheiro para escrita com o nome do modelo
    with open(f"{model_name}.txt", "w") as f:
        f.write(model_name)
        f.write("\n")

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    fake_data_path = os.sep.join(["original_data", "fake"])
    real_data_path = os.sep.join(["original_data", "real"])

    fake_images = [os.path.join(fake_data_path, file) for file in os.listdir(fake_data_path) if file.endswith(".nii.gz")]
    real_images = [os.path.join(real_data_path, file) for file in os.listdir(real_data_path) if file.endswith(".nii.gz")]

    # Labels: 0 for fake, 1 for real
    correct_labels = [0]*50 + [1]*50
    images = fake_images + real_images

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])

    # Define image dataset, data loader
    check_ds = ImageDataset(image_files=images, labels=correct_labels, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    im, label = monai.utils.misc.first(check_loader)
    print(type(im), im.shape, label)
    with open(f"{model_name}.txt", "a") as f:
        f.write(f"{type(im)}, {im.shape}, {label}\n")

    # create a training data loader
    train_indices = list(range(35)) + list(range(50, 85))
    val_indices = list(range(35, 50)) + list(range(85, 100))


    train_ds = ImageDataset(image_files=[images[i] for i in train_indices], 
                            labels=[correct_labels[i] for i in train_indices], 
                            transform=train_transforms)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())


    # create a validation data loader
    val_ds = ImageDataset(image_files=[images[i] for i in val_indices], 
                          labels=[correct_labels[i] for i in val_indices], 
                          transform=val_transforms)

    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "DenseNet121":
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    elif model_name == "ResNet":
        model = monai.networks.nets.ResNet(block='basic', layers=[2, 2, 2, 2], block_inplanes=[64, 128, 256, 512], spatial_dims=3, n_input_channels=1, num_classes=1).to(device)
    else:
        raise ValueError(f"Model '{model_name}' não reconhecido.")

    loss_function = torch.nn.BCEWithLogitsLoss() 

    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        #write to model file
        with open(f"{model_name}.txt", "a") as f:
            f.write("-" * 10)
            f.write(f"epoch {epoch + 1}/{5}\n")
        model.train()
        epoch_loss = 0
        step = 0
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
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            #write to model file
            with open(f"{model_name}.txt", "a") as f:
                f.write(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}\n")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        #write to model file
        with open(f"{model_name}.txt", "a") as f:
            f.write(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}\n")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                probs = []
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    val_probs = F.sigmoid(val_outputs)
                    probs.append(val_probs.cpu().numpy())
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                print(f"Predicted Probabilities: {probs}")

                writer.add_scalar("val_accuracy", metric, epoch + 1)
                #write to model file
                with open(f"{model_name}.txt", "a") as f:
                    f.write(
                        f"current epoch: {epoch + 1} current accuracy: {metric:.4f} best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}\n"
                    )
                    f.write(f"Predicted Probabilities: {probs}\n")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
