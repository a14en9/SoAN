#!/usr/bin/env python
import argparse
import pathlib
import yaml
import json
from addict import Dict
import torch
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from utils.data_loader import fetch_loaders
from utils.frame import Framework, IOStream
from utils.metrics import diceloss, bce_diceloss
from torch.utils.tensorboard import SummaryWriter
import utils.train as tr


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main.py outputs'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp models/HarDMSEG.py outputs' + '/' + args.exp_name + '/' + 'HarDMSEG.py.backup')
    os.system('cp utils/data_loader.py outputs' + '/' + args.exp_name + '/' + 'data_loader.py.backup')

def train(args, io):
    data_dir = args.data_dir
    model_dir = args.model_dir
    conf = Dict(yaml.safe_load(open(args.train_yaml, "r")))
    loss_type = args.loss_type
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = args.device
    # if device is not None:
    #     device = torch.device(device)
    args = Dict({
        "batch_size": args.batch_size,
        "exp_name": args.exp_name,
        "epochs": args.epochs,
        "save_every": args.save_every
    })
    loaders = fetch_loaders(data_dir, args.batch_size, shuffle=True)
    # test_loader = fetch_loaders(data_dir, args.batch_size, folder='test', shuffle=False)
    # if input mask dimension different than outchannels
    outchannels = conf.model_opts.args.outchannels
    # y_channels = [y.shape[-1] for _, y in loaders["test"]][0]   ### y_channels = test loader if there isn't validation set
    # if y_channels != outchannels:
    #     raise ValueError("Output dimension is different from model outchannels.")
    # TODO: try to have less nested if/else
    # get dice loss
    if loss_type == "bce_dice":
        loss_weight = [0.6, 0.4]  # clean ice, debris, background [0.7, 0.3]
        label_smoothing = 0.2
        loss_fn = bce_diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                               outchannels=outchannels, label_smoothing=label_smoothing)
    else:
        loss_weight = [0.6, 0.4]
        label_smoothing = 0.2
        loss_fn = diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                           outchannels=outchannels, label_smoothing=label_smoothing)
    # Try to load the models
    frame = Framework(
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts,
        loss_fn=loss_fn,
        device=device
    )

    # print(str(frame))
    # Setup logging
    writer = SummaryWriter(f"{model_dir}/{args.exp_name}/logs/")
    writer.add_text("Arguments", json.dumps(vars(args)))
    writer.add_text("Configuration Parameters", json.dumps(conf))
    out_dir = f"{model_dir}/{args.exp_name}/models/"
    mask_names = conf.log_opts.mask_names

    best_test_iou = 0
    for epoch in range(args.epochs):

        # train loop
        loss_d = {}
        loss_d["train"], train_metrics = tr.train_epoch(loaders["train"], frame, conf.metrics_opts)
        tr.log_metrics(writer, train_metrics, loss_d["train"], epoch, mask_names=mask_names)
        if (epoch + 1) % args.save_every == 0:
            tr.log_images(writer, frame, next(iter(loaders["train"])), epoch)
        outstr = 'Train Epoch %d, Loss: %.6f, Train Glacial Lake IoU: %.6f, Train Background IoU: %.6f' % (epoch, loss_d['train'], train_metrics['IoU'][0], train_metrics['IoU'][1])
        io.cprint(outstr)
        # Validation loop
        loss_d["val"], val_metrics = tr.validate(loaders["val"], frame, conf.metrics_opts)

        tr.log_metrics(writer, val_metrics, loss_d["val"], epoch, "val", mask_names=mask_names)
        # if (epoch + 1) % args.save_every == 0:
        #     tr.log_images(writer, frame, next(iter(loaders["val"])), epoch, "val")
        outstr = 'Val   Epoch %d, Loss: %.6f, Val   Glacial Lake IoU: %.6f, Val   Background IoU: %.6f' % (epoch, loss_d['val'], val_metrics['IoU'][0], val_metrics['IoU'][1])
        io.cprint(outstr)
        # Save model
        writer.add_scalars("Loss", loss_d, epoch)
        if (epoch + 1) % args.save_every == 0:
            frame.save(out_dir, epoch)
        if np.mean(val_metrics['IoU'][0].cpu().detach().numpy()) >= best_test_iou:
            best_test_iou = np.mean(val_metrics['IoU'][0].cpu().detach().numpy())
            frame.save(out_dir, "optimal")
            tr.log_images(writer, frame, next(iter(loaders["val"])), epoch, "val")

    # frame.save(out_dir, "final")
    writer.close()

def test(args, io):
    data_dir = args.data_dir
    model_dir = args.model_dir
    conf = Dict(yaml.safe_load(open(args.train_yaml, "r")))
    outchannels = conf.model_opts.args.outchannels
    loaders = fetch_loaders(data_dir, args.batch_size, shuffle=True)
    device = torch.device("cuda" if args.cuda else "cpu")
    # TODO: try to have less nested if/else
    loss_type = args.loss_type
    # get dice loss
    if loss_type == "bce_dice":
        loss_weight = [0.6, 0.4]  # clean ice, debris, background [0.7, 0.3]
        label_smoothing = 0.2
        loss_fn = bce_diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                               outchannels=outchannels, label_smoothing=label_smoothing)
    else:
        loss_weight = [0.6, 0.4]
        label_smoothing = 0.2
        loss_fn = diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                           outchannels=outchannels, label_smoothing=label_smoothing)
    # Try to load the models
    frame = Framework(
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts,
        loss_fn=loss_fn,
        device=device
    )
    frame.model.load_state_dict(torch.load(args.saved_model_dir))
    model = frame.model.to(device)
    model = model.eval()
    t_metrics = {}
    # channel_first = lambda x: x.permute(0, 3, 1, 2)

    slices_dir = f"{data_dir}/test/*img*"
    pred_dir = f"{model_dir}/{args.exp_name}/preds/"

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    slices = glob.glob(slices_dir)
    total_inference_time = 0
    for s in slices:
        filename = s.split("/")[-1].replace("npy", "png")
        inp_np = np.load(s)
        start = time.time()
        nan_mask = np.isnan(inp_np[:, :, :]).any(axis=2)
        inp_tensor = torch.from_numpy(np.expand_dims(np.transpose(inp_np, (2, 0, 1)), axis=0))
        inp_tensor = inp_tensor.to(device)
        output = model(inp_tensor)
        output_np = output.detach().cpu().numpy()
        output_np = np.transpose(output_np[0], (1, 2, 0))
        output_np = np.argmax(output_np, axis=2)
        output_np[nan_mask] = 3
        total_inference_time += (time.time() - start)
        average_inference_time = total_inference_time/len(slices)
        # plt.imsave(f"{pred_dir}{filename}", output_np, vmin=0, vmax=3)
        plt.imsave(f"{pred_dir}{filename}", 1 - output_np, cmap='gray')

    print(f"Total Inference Time               : {total_inference_time}")
    print(f"Average Inference Time for an Image: {average_inference_time}")

    for x, y in loaders["test"]:
        x = x.permute(0, 3, 1, 2).to(device)
        y_hat = model(x).permute(0, 2, 3, 1)
        y_hat = frame.segment(y_hat)
        metrics_ = frame.metrics(y_hat, y, conf.metrics_opts)
        tr.update_metrics(t_metrics, metrics_)
    tr.agg_metrics(t_metrics)
    outstr = 'Test Glacial Lake IoU:        %.6f, Test Background IoU:        %.6f, \n' \
             'Test Glacial Lake Pixel_Acc:  %.6f, Test Background Pixel_Acc:  %.6f, \n' \
             'Test Glacial Lake Precision:  %.6f, Test Background Precision:  %.6f, \n' \
             'Test Glacial Lake Recall:     %.6f, Test Background Recall:     %.6f, \n' \
             'Test Glacial Lake Dice Score: %.6f, Test Background Dice Score: %.6f, \n' \
             'Test Glacial Lake F1 Score:   %.6f, Test Background F1 Score:   %.6f, \n' \
             'Test Glacial Lake F2 Score:   %.6f, Test Background F2 Score:   %.6f,' \
             % (t_metrics['IoU'][0], t_metrics['IoU'][1],
                t_metrics['pixel_acc'][0], t_metrics['pixel_acc'][1],
                t_metrics['precision'][0], t_metrics['precision'][1],
                t_metrics['recall'][0], t_metrics['recall'][1],
                t_metrics['dice'][0], t_metrics['dice'][1],
                t_metrics['f1_score'][0], t_metrics['f1_score'][1],
                t_metrics['f2_score'][0], t_metrics['f2_score'][1])
    io.cprint(outstr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw tiffs into slices")
    parser.add_argument("--data_dir", type=str, default="patches/splits/")
    parser.add_argument("--model_dir", type=str, default="outputs/")
    parser.add_argument("--saved_model_dir", type=str, default="outputs/HarDMSEG/models/model_optimal.pt")
    parser.add_argument("--train_yaml", type=str, default="conf/train_hardmseg.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--exp_name", type=str, default="HarDMSEG")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--loss_type", type=str, default="dice")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False, help='evaluate the model')
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    _init_()


    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    # test(args, io)
    if not args.eval:
        train(args, io)
    else:
        test(args, io)