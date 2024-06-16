"""
Adversarial training on the traversability dataset
"""

from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import attacks
from pptx import Presentation
from pptx.util import Inches
import pandas as pd
from datetime import date

from torch.utils import data
from datasets import IndoorTrav
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import matplotlib.pyplot as plt

from trav_finetune import per_image_metric


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/home/qiyuan/2023spring/segmentation_indoor_images',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='trav',
                        choices=['voc', 'cityscapes', 'trav'], help='Name of dataset')
    parser.add_argument("--scenes", type=list, default=['elb', 'erb', 'uc', 'woh'],
                        choices=['elb', 'erb', 'uc', 'nh', 'woh'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results\"")
    # parser.add_argument("--total_itrs", type=int, default=3e4,
    #                     help="total iter number (default: 3e4)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="total number of epochs (default: 1e2)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=480)

    parser.add_argument("--ckpt", default='checkpoints/last_deeplabv3plus_resnet101_trav_os16.pth',
                        type=str, help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--eps", type=float, default=0.01,
                        help="PGD attack epsilon")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="PGD attack alpha")
    parser.add_argument("--pgd_iter", type=int, default=100,
                        help="PGD attack iterations")
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: cross_entropy)")
    parser.add_argument("--gpu_id", type=str, default='0,1',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=444,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'trav':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.5174, 0.4857, 0.5054],
                            std=[0.2726, 0.2778, 0.2861]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.5174, 0.4857, 0.5054],
                            std=[0.2726, 0.2778, 0.2861]),
        ])

        train_dst = IndoorTrav(opts.data_root, 'train', opts.scenes,
                               transform=train_transform)
        val_dst = IndoorTrav(opts.data_root, 'val', opts.scenes,
                             transform=val_transform)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, epoch, clean=False):
    """
    clean: bool, evaluate on clean or adv examples
    """
    metrics.reset()
    perimage_metrics = StreamSegMetrics(opts.num_classes)
    denorm = utils.Denormalize(mean=[0.5174, 0.4857, 0.5054],
                               std=[0.2726, 0.2778, 0.2861])
    img_id = 0

    progress_bar = tqdm(loader)
    for i, (x_clean, y_true, filenames) in enumerate(progress_bar):
        x_clean = x_clean.to(device, dtype=torch.float32)
        y_true = y_true.to(device, dtype=torch.uint8)
        
        y_pred_clean, _ = model(x_clean)
        if clean:
            delta1 = torch.zeros_like(x_clean)
        else:
            delta1 = attacks.pgd(model, x_clean, y_true, device, epsilon=opts.eps, alpha=opts.alpha, num_iter=opts.pgd_iter)

        x_adv = x_clean.float() + delta1.float()
        y_pred_adv, _ = model(x_adv)
        y_pred_adv_np = y_pred_adv.detach().max(dim=1)[1].cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_clean_np = y_pred_clean.detach().max(dim=1)[1].cpu().numpy()
        metrics.update(y_true_np, y_pred_adv_np)

        if opts.save_val_results:
            for j in range(len(x_clean)):
                image = x_clean[j].detach().cpu().numpy()
                delta_np = np.sum(delta1[j].detach().cpu().numpy(), axis=0)
                total_pertub = np.sum(np.abs(delta_np))
                target = y_true_np[j]  # y_true
                pred = y_pred_adv_np[j]  # adv y_pred
                output = y_pred_clean_np[j]  # clean y_pred
                adv_iou = per_image_metric(perimage_metrics, target, pred)
                clean_iou = per_image_metric(perimage_metrics, target, output)
                adv_img = x_adv[j].detach().cpu().numpy()
                effect = (clean_iou['Mean IoU'] - adv_iou['Mean IoU'])/total_pertub

                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)

                adv_img = (denorm(adv_img) * 255).transpose(1, 2, 0).astype(np.uint8)

                img_id += 1

        progress_bar.set_description(f'{i}/{len(loader)}')

    score = metrics.get_results()
    print(f'iter: {epoch}; {score}')

    return score


def kl_div_loss(logits_q, logits_p, T):
    assert logits_p.size() == logits_q.size()
    logits_q = logits_q.view(logits_q.size()[0], -1)
    logits_p = logits_p.view(logits_p.size()[0], -1)
    b, c = logits_p.size()
    p = nn.Softmax(dim=1)(logits_p / T)
    q = nn.Softmax(dim=1)(logits_q / T)
    epsilon = 1e-8
    _p = (p + epsilon * torch.ones(b, c).cuda()) / (1.0 + c * epsilon)
    _q = (q + epsilon * torch.ones(b, c).cuda()) / (1.0 + c * epsilon)
    return (T ** 2) * torch.mean(torch.sum(_p * torch.log(_p / _q), dim=1))


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'trav':
        opts.num_classes = 2

    wandb.init(
        # set the wandb project where this run will be logged
        project="SegPGD",
        # track hyperparameters and run metadata
        config=opts
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {opts.gpu_id}")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.epochs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    mse_criterion = nn.MSELoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        # model.classifier.classifier[-1] = nn.Conv2d(256, 2, 1)
        model = nn.DataParallel(model)
        model.to(device)

        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    # vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
    #                                   np.int32) if opts.enable_vis else None  # sample idxs for visualization
    # denorm = utils.Denormalize(mean=[0.5174, 0.4857, 0.5054], std=[0.2726, 0.2778, 0.2861])  # denormalization for ori images

    print('\n[ Train epoch: %d ]' % opts.epochs)
    model.train()

    cur_itrs = 0

    for e in range(opts.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch: {e}/{opts.epochs}")
        for i, (images, labels, filenames) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            clean_outputs, clean_features = model(images)
            delta = attacks.pgd(model, images, labels, device, epsilon=opts.eps, alpha=opts.alpha, num_iter=opts.pgd_iter)
            x_adv = images.float() + delta.float()
            adv_outputs, adv_features = model(x_adv)
            loss = criterion(adv_outputs, labels)
            # h_loss = mse_criterion(adv_features['out'], clean_features['out'])
            h_loss = kl_div_loss(adv_features['out'], clean_features['out'].detach(), 1)
            total_loss = loss + h_loss
            total_loss.backward()

            optimizer.step()
            progress_bar.set_postfix({"total_loss": total_loss.item(), 'loss': loss.item(), 'h_loss': h_loss.item()})
            wandb.log({"total_loss": total_loss.item(), 'loss': loss.item(), 'h_loss': h_loss.item()})
        scheduler.step()

        if e+1 % opts.val_interval == 0:
            # save_ckpt(f'checkpoints/latest_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth')
            print("validation...")
            model.eval()
            val_score = validate(
                opts, model, val_loader, device, metrics, cur_itrs,
                False)
            wandb.log({'mIoU': val_score['Mean IoU']})
            print(metrics.to_str(val_score))
            if val_score['Mean IoU'] > best_score:  # save best model
                best_score = val_score['Mean IoU']
                save_ckpt('checkpoints/best_%s_%s_os%d.pt' %
                            (opts.model, opts.dataset, opts.output_stride))

            model.train()
            cur_itrs += 1

    val_score = validate(
        opts, model, val_loader, device, metrics, cur_itrs,
        False)
    wandb.log({'mIoU': val_score['Mean IoU']})
    print(metrics.to_str(val_score))
    save_ckpt(f'checkpoints/hloss_{opts.model}_{opts.dataset}_os{opts.output_stride}_{date.today()}.pt')
    wandb.finish()


if __name__ == '__main__':
    main()
