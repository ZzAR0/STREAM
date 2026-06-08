"""Train CoViAR baselines or the STREAM fusion model."""

import os
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision

from dataset import CoviarDataSet
from model import Model
from train_options import parser
from transforms import GroupCenterCrop, GroupScale, StreamGroupCenterCrop, StreamGroupScale


SAVE_FREQ = 40
PRINT_FREQ = 20
best_prec1 = 0


def move_to_cuda(x):
    if isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    return x.cuda(non_blocking=True).contiguous()


def batch_size_of(x):
    if isinstance(x, dict):
        return next(iter(x.values())).size(0)
    return x.size(0)


def aggregate_segments_if_needed(output, target):
    if output.size(0) == target.size(0):
        return output
    return output.view((-1, args.num_segments) + output.size()[1:]).mean(dim=1)


def main():
    global args, best_prec1
    parser.add_argument("--resume", default="", type=str, help="checkpoint path")
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--save-dir", default="./checkpoints", type=str)
    parser.add_argument("--lr-scheduler", default="step", choices=["step", "plateau"], type=str)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--factor", default=0.3, type=float)
    parser.add_argument("--min-lr", default=1e-7, type=float)
    parser.add_argument("--eps", default=1e-4, type=float)
    parser.add_argument("--stream-dropout", default=0.1, type=float)
    parser.add_argument("--warp-direction", default=1.0, type=float)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Training arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    if args.data_name == "ucf101":
        num_class = 101
    elif args.data_name == "hmdb51":
        num_class = 51
    elif args.data_name == "try":
        num_class = 2
    else:
        raise ValueError("Unknown dataset " + args.data_name)

    model = Model(
        num_class,
        args.num_segments,
        args.representation,
        base_model=args.arch,
        stream_dropout=args.stream_dropout,
        warp_direction=args.warp_direction,
    )
    print(model)

    if args.representation == "stream":
        val_transform = torchvision.transforms.Compose([
            StreamGroupScale(int(model.scale_size)),
            StreamGroupCenterCrop(model.crop_size),
        ])
    else:
        val_transform = torchvision.transforms.Compose([
            GroupScale(int(model.scale_size)),
            GroupCenterCrop(model.crop_size),
        ])

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=model.get_augmentation(),
            is_train=True,
            accumulate=(not args.no_accumulation),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(args.representation == "stream"),
    )

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=val_transform,
            is_train=False,
            accumulate=(not args.no_accumulation),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    cudnn.benchmark = True

    optimizer = torch.optim.Adam(make_parameter_groups(model), weight_decay=args.weight_decay, eps=0.001)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.factor,
            patience=args.patience,
            min_lr=args.min_lr,
            eps=args.eps,
        )

    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        args.start_epoch = checkpoint.get("epoch", 0)
        best_prec1 = checkpoint.get("best_prec1", 0)
        state_dict = checkpoint["state_dict"]
        is_legacy_stream = args.representation == "stream" and not any(
            k.startswith("module.stream_model.temporal_model.") or k.startswith("stream_model.temporal_model.")
            for k in state_dict.keys()
        )
        if is_legacy_stream:
            print("Warning: resuming a legacy STREAM checkpoint without MS-TCN.")
            print("Warning: temporal module will stay randomly initialized.")
        model.load_state_dict(checkpoint["state_dict"], strict=not is_legacy_stream)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

    for epoch in range(args.start_epoch, args.epochs):
        if args.lr_scheduler == "step":
            cur_lr = adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)
        else:
            cur_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_prec1 = train(train_loader, model, criterion, optimizer, epoch, cur_lr)

        latest_state = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "representation": args.representation,
            "stream_dropout": args.stream_dropout,
            "warp_direction": args.warp_direction,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        save_named_checkpoint(latest_state, epoch + 1, kind="lastest")

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, val_loss = validate(val_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            latest_state["best_prec1"] = best_prec1
            latest_state["state_dict"] = model.state_dict()
            latest_state["optimizer"] = optimizer.state_dict()
            latest_state["scheduler"] = scheduler.state_dict() if scheduler is not None else None
            save_named_checkpoint(latest_state, epoch + 1, kind="lastest")
            if is_best:
                save_named_checkpoint(latest_state, epoch + 1, kind="best")
            if scheduler is not None:
                scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1} done: train_loss={train_loss:.4f}, "
            f"train_prec1={train_prec1:.3f}, best_prec1={best_prec1:.3f}",
            flush=True,
        )


def make_parameter_groups(model):
    params = []
    for key, value in model.named_parameters():
        decay_mult = 0.0 if "bias" in key else 1.0
        if args.representation == "stream":
            if "classifier" in key or "gate" in key or "_proj" in key:
                lr_mult = 1.0
            else:
                lr_mult = 0.01
        elif (
            "module.base_model.conv1" in key or "module.base_model.bn1" in key or "data_bn" in key
        ) and args.representation in ["mv", "residual"]:
            lr_mult = 0.1
        elif ".fc." in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.01
        params.append({"params": value, "lr": args.lr * lr_mult, "lr_mult": lr_mult, "decay_mult": decay_mult})
    return params


def train(train_loader, model, criterion, optimizer, epoch, cur_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    end = time.time()

    for i, (input_data, target) in enumerate(train_loader):
        if input_data is None:
            continue
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True).long().contiguous()
        input_data = move_to_cuda(input_data)

        output = model(input_data)
        output = aggregate_segments_if_needed(output, target)
        loss = criterion(output, target)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        bs = batch_size_of(input_data)
        losses.update(loss.item(), bs)
        top1.update(prec1.item(), bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % PRINT_FREQ == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}], lr: {cur_lr:.7f}\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
            )
    return losses.avg, top1.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            if input_data is None:
                continue
            target = target.cuda(non_blocking=True).long().contiguous()
            input_data = move_to_cuda(input_data)

            output = model(input_data)
            output = aggregate_segments_if_needed(output, target)
            loss = criterion(output, target)

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            bs = batch_size_of(input_data)
            losses.update(loss.item(), bs)
            top1.update(prec1.item(), bs)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % PRINT_FREQ == 0:
                print(
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
                )

    print(f"Testing Results: Prec@1 {top1.avg:.3f} Loss {losses.avg:.5f}")
    return top1.avg, losses.avg


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    inputs, targets = zip(*batch)
    targets = torch.tensor([int(t) for t in targets], dtype=torch.long)
    if isinstance(inputs[0], dict):
        stacked = {k: torch.stack([x[k] for x in inputs]).contiguous() for k in inputs[0].keys()}
    else:
        stacked = torch.stack(inputs).contiguous()
    return stacked, targets


def save_named_checkpoint(state, epoch, kind):
    if args.representation == "stream":
        if args.arch == "resnet18":
            model_tag = "resnet[18+18]"
        else:
            arch_id = args.arch.replace("resnet", "")
            model_tag = "resnet[{}+{}]".format(arch_id, arch_id)
    else:
        model_tag = args.arch
    filename = os.path.join(args.save_dir, f"{model_tag}_{epoch}_{kind}.pth")
    torch.save(state, filename)
    alias = os.path.join(args.save_dir, f"{model_tag}_{kind}.pth")
    torch.save(state, alias)
    print(f"=> checkpoint saved to {filename}")
    print(f"=> checkpoint alias saved to {alias}")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group["lr_mult"]
        param_group["weight_decay"] = wd * param_group["decay_mult"]
    return lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
