# train_cnn_with_preprocessing.py
import torch
import matplotlib.pyplot as plt
from train.cnn_training import (
    cnn_dataloaders,
    train_cnn,
    test_cnn
)

from train.vit_training import (
    vit_dataloaders,
    train_vit,
    test_vit
)

from train.swin_training import (
    swin_dataloaders,
    train_swin,
    test_swin
)

from train.mlp_training import (
    mlp_dataloaders,
    train_mlp,
    test_mlp
)


def load_filelist(txt_path):
    with open(txt_path, "r") as f:
        lines = f.read().strip().splitlines()
    return lines

def main():

    train_files = load_filelist("dataset_processed/train.txt")
    val_files = load_filelist("dataset_processed/validation.txt")
    test_files = load_filelist("dataset_processed/test.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    batch_size = 64
    epochs = 50
    learning_rate = 1e-4
    num_workers = 26

    # # CNN
    # cnn_train_loader, cnn_val_loader, cnn_test_loader, cnn_class_names = cnn_dataloaders(
    #     train_files, val_files, test_files,
    #     batch_size=batch_size,
    #     img_size=224,
    #     num_workers=num_workers
    # )
    #
    # cnn_model, cnn_train_loss_history, cnn_val_acc_history = train_cnn(cnn_train_loader, cnn_val_loader, cnn_class_names,
    #                   lr=learning_rate, epochs=epochs,
    #                   device=device,
    #                   save_path="saved_model/best_cnn.pth")
    #
    # # ViT
    # vit_train_loader, vit_val_loader, vit_test_loader, vit_class_names = vit_dataloaders(
    #     train_files, val_files, test_files,
    #     batch_size=64,
    #     img_size=224,
    #     num_workers=num_workers,
    #     upper_limit=5000,
    #     lower_limit=2000
    # )
    #
    #
    # vit_model, vit_train_loss_history, vit_val_acc_history = train_vit(vit_train_loader, vit_val_loader, vit_class_names,
    #                   model_name="vit_base_patch16_224",
    #                   pretrained=True,
    #                   lr=learning_rate,
    #                   num_epochs=epochs,
    #                   device=device,
    #                   save_path="saved_model/best_vit.pth")

    # swin
    swin_train_loader, swin_val_loader, swin_test_loader, swin_class_names = swin_dataloaders(
        train_files, val_files, test_files,
        batch_size=64,
        img_size=224,
        num_workers=26,
        upper_limit=100000,
        lower_limit=0
    )

    print(swin_class_names)
    swin_model, swin_train_loss_history, swin_val_acc_history = train_swin(swin_train_loader, swin_val_loader, swin_class_names,
                       model_name="swin_base_patch4_window7_224",
                       pretrained=True,
                       lr=learning_rate,
                       num_epochs=epochs,
                       device=device,
                       save_path="saved_model/best_swin.pth")

    ## MLP
    mlp_train_loader, mlp_val_loader, mlp_test_loader, mlp_class_names = mlp_dataloaders(
        train_files, val_files, test_files,
        batch_size=batch_size,
        img_size=224,
        num_workers=num_workers,
        upper_limit=5000,
        lower_limit=2000
    )

    print("train mlp")
    mlp_model, mlp_train_loss_history, mlp_val_acc_history = train_mlp(mlp_train_loader, mlp_val_loader, mlp_class_names,
                            model_name="mixer_b16_224",
                            pretrained=True,
                            lr=learning_rate,
                            num_epochs=epochs,
                            device=device,
                            save_path="saved_model/best_mlp_mixer.pth")


    # Test
    # cnn_test_loss, cnn_test_acc = test_cnn(cnn_model, cnn_test_loader, device)
    # print("CNN Accuracy:", cnn_test_acc)
    #
    # vit_test_loss, vit_test_acc = test_vit(vit_model, vit_test_loader, device)
    # print("Visual Transformer Accuracy:", vit_test_acc)

    swin_test_loss, swin_test_acc = test_swin(swin_model, swin_test_loader, device)
    print("Swing Transformer Accuracy:", swin_test_acc)

    mlp_test_loss, mlp_test_acc = test_mlp(mlp_model, mlp_test_loader, device)
    print(f"MLP Accuracy = {mlp_test_acc:.4f}")

    # train loss vs epoch
    epochs = range(1, len(swin_train_loss_history) + 1)
    plt.plot(epochs, cnn_train_loss_history, label="CNN")
    plt.plot(epochs, vit_train_loss_history, label="ViT")
    plt.plot(epochs, swin_train_loss_history, label="Swin")
    plt.plot(epochs, mlp_train_loss_history, label="MLP")

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Train Loss vs. Epochs")
    plt.legend()

    plt.savefig("loss_compare.png")
    plt.show()

    # acc vs epoch
    epochs = range(1, len(swin_val_acc_history) + 1)
    plt.plot(epochs, cnn_val_acc_history, label="CNN")
    plt.plot(epochs, vit_val_acc_history, label="ViT")
    plt.plot(epochs, swin_val_acc_history, label="Swin")
    plt.plot(epochs, mlp_val_acc_history, label="MLP")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Val Accuracy vs. Epochs")
    plt.legend()

    plt.savefig("results/accuracy_compare1.png")
    plt.show()


if __name__ == "__main__":
    main()


