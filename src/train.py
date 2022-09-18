from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from albumentations.pytorch import ToTensorV2
from warmup_scheduler import GradualWarmupScheduler
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from sklearn.model_selection import KFold
from metrics import accuracy_metric
from dataset import BrainDataset
from config import config as cfg_file

import timm
import albumentations as A
import wandb
import torch
import json
import numpy as np
import os

def get_dataset(root_dir, is_train=True):
    if is_train:
        transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Rotate(limit=180, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    elif not is_train:
        transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                ToTensorV2(),
            ]
        )
    train_dataset = BrainDataset(root_dir, is_train=is_train, transform=transform)
    return train_dataset

def make_dataloader(dataset, batch_size):
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    return loader

def make_oversamp_loader(dataset,batch_size):
    labels = []
    for image, label in dataset:
        labels.append(label.item())
    values, counts = np.unique(labels, return_counts=True)
    class_weights = [1/count for count in counts.tolist()]

    sample_weights = [0] * len(dataset)

    for idx, (image, labels) in enumerate(dataset):
        class_weight = class_weights[labels.item()]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
    print("[x] Create Oversampling data loader successful")
    return loader

def create_model(config):
    model = timm.create_model(config.backbone, pretrained=config.pretrained, 
                                         in_chans=config.in_channels, num_classes=config.classes)
    return model

def train(model, train_loader, val_loader, criterion, metrics, optimizer, config, fold, scheduler=None):
    wandb.watch(model, criterion, log="all")
    train_epoch = TrainEpoch(
                        model, 
                        loss=criterion,
                        metrics= metrics,
                        optimizer=optimizer,
                        device=DEVICE,
                        verbose=True)

    val_epoch = ValidEpoch(
                    model, 
                    loss=criterion,
                    metrics=metrics,
                    device=DEVICE,
                    verbose=True)

    best_accuracy = 0
    for epoch in range(config.epoch):
        print(f"[X] Epoch {epoch + 1} :")
        if scheduler is not None:
            scheduler.step(epoch)

        train_logs = train_epoch.run(train_loader)
        wandb.log({"epoch": epoch, "train_loss": train_logs['Cross_Entropy_loss'], "train_accuracy": train_logs['accuracy']})

        val_logs = val_epoch.run(val_loader)
        wandb.log({"epoch": epoch, "val_loss": val_logs['Cross_Entropy_loss'], "val_accuracy": val_logs['accuracy']})

        val_accuracy = val_logs['accuracy']
        if val_accuracy > best_accuracy:
            print(f"[X] Val acc improved from {best_accuracy} to {val_accuracy}, save weight")
            best_accuracy = val_accuracy
            save_name = os.path.join(cfg_file['weight_path'], f"{config.backbone}_fold_{fold+1}.pth")
            torch.save(model.state_dict(), save_name)

def test(model, test_loader, criterion, metrics):
    val_epoch = ValidEpoch(
                    model, 
                    loss=criterion,
                    metrics=metrics,
                    device=DEVICE,
                    verbose=True)

    test_logs = val_epoch.run(test_loader)
    wandb.log({"test_accuracy": test_logs['accuracy']})

def summarize(model, train_loader, val_loader, test_loader, criterion, metrics):
    val_epoch = ValidEpoch(
                    model, 
                    loss=criterion,
                    metrics=metrics,
                    device=DEVICE,
                    verbose=False)

    train_eval = val_epoch.run(train_loader)
    val_eval = val_epoch.run(val_loader)
    test_eval = val_epoch.run(test_loader)

    data = [
        ['Train', train_eval['accuracy']],
        ['Validation', val_eval['accuracy']],
        ['Test', test_eval['accuracy']],
    ]
            
    columns=["Eval on", "Accuracy"]
    sum_table = wandb.Table(data=data, columns=columns)
    wandb.log({"Evaluation summarize": sum_table})

    print(f"train acc : {train_eval['accuracy']}, val acc : {val_eval['accuracy']}, test : {test_eval['accuracy']}")

def trainKfold(hyperparams, K=5):
    train_dataset = get_dataset(cfg_file['dataset_root'])
    test_dataset = get_dataset(cfg_file['dataset_root'], False)

    kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):

        exp_name = f"{hyperparams['backbone']}_fold_{fold+1}"
        with wandb.init(project="MRI-Brain-tumor-cls", name=exp_name, config=hyperparams):
            config = wandb.config

            train_subsampler = Subset(train_dataset,train_ids)
            val_subsampler = Subset(train_dataset,val_ids)

            if config.oversampling:
                train_loader = make_oversamp_loader(train_subsampler, config.batch_size)
            else:
                train_loader = make_dataloader(train_subsampler, config.batch_size)
            val_loader = make_dataloader(val_subsampler, config.batch_size)
            test_loader = make_dataloader(test_dataset, hyperparams['batch_size'])

            with torch.cuda.device('cuda:0'):
                torch.cuda.empty_cache()

            model = create_model(config)
            loss = torch.nn.CrossEntropyLoss()
            loss.__name__ = 'Cross_Entropy_loss'
            metrics = [accuracy_metric()]
            optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
            
            if config.lr_warmup:           
                scheduler_steplr = ExponentialLR(optimizer, gamma=0.97)
                scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=scheduler_steplr)
            else:
                scheduler = None
            
            optimizer.zero_grad()
            optimizer.step()

            print("[X] Start training " + exp_name)
            train(model, train_loader, val_loader, loss, metrics, optimizer, config, fold, scheduler)
    
            print("[X] Load best weight for evaluation")
            weight_path = os.path.join(cfg_file['weight_path'], f"{config.backbone}_fold_{fold+1}.pth")
            weight = torch.load(weight_path)
            model.load_state_dict(weight)
            model.eval()

            test(model, test_loader, loss, metrics)
            summarize(model, train_loader, val_loader, test_loader, loss, metrics)

            del model, optimizer
        
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    for backbone in ['vgg16', 'resnet18', 'resnet34', 'resnet50', 'seresnet18', 'seresnet34', 'seresnet50', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'resnext50_32x4d','regnety_004', 'regnety_006', 'regnety_008']:
        hyperparams = dict(
                backbone=backbone,
                epoch=int(cfg_file['epoch']),
                batch_size=int(cfg_file['batch_size']),
                in_channels=int(cfg_file['in_channels']),
                classes=int(cfg_file['classes']),
                pretrained=cfg_file['pretrained'],
                lr_warmup=json.loads(cfg_file['lr_warmup'].lower()),
                lr=float(cfg_file['lr']),
                oversampling=json.loads(cfg_file['oversampling'].lower())
                )

        trainKfold(hyperparams, K=int(cfg_file['fold']))