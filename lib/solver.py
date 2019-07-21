import os
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import sys
sys.path.append(".")
from lib.utils import decode_eta
from lib.config import CONF
from eval import compute_acc, compute_miou

ITER_REPORT_TEMPLATE = """
----------------------iter: [{global_iter_id}/{total_iter}]----------------------
[loss] train_loss: {train_loss}
[sco.] train_point_acc: {train_point_acc}
[sco.] train_point_acc_per_class: {train_point_acc_per_class}
[sco.] train_voxel_acc: {train_voxel_acc}
[sco.] train_voxel_acc_per_class: {train_voxel_acc_per_class}
[sco.] train_point_miou: {train_point_miou}
[sco.] train_voxel_miou: {train_voxel_miou}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
------------------------summary------------------------
[train] train_loss: {train_loss}
[train] train_point_acc: {train_point_acc}
[train] train_point_acc_per_class: {train_point_acc_per_class}
[train] train_voxel_acc: {train_voxel_acc}
[train] train_voxel_acc_per_class: {train_voxel_acc_per_class}
[train] train_point_miou: {train_point_miou}
[train] train_voxel_miou: {train_voxel_miou}
[val]   val_loss: {val_loss}
[val]   val_point_acc: {val_point_acc}
[val]   val_point_acc_per_class: {val_point_acc_per_class}
[val]   val_voxel_acc: {val_voxel_acc}
[val]   val_voxel_acc_per_class: {val_voxel_acc_per_class}
[val]   val_point_miou: {val_point_miou}
[val]   val_voxel_miou: {val_voxel_miou}
"""

BEST_REPORT_TEMPLATE = """
-----------------------------best-----------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[sco.] point_acc: {point_acc}
[sco.] point_acc_per_class: {point_acc_per_class}
[sco.] voxel_acc: {voxel_acc}
[sco.] voxel_acc_per_class: {voxel_acc_per_class}
[sco.] point_miou: {point_miou}
[sco.] voxel_miou: {voxel_miou}
"""

class Solver():
    def __init__(self, model, dataloader, criterion, optimizer, batch_size, stamp, is_wholescene=True, decay_step=10, decay_factor=0.7):
        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.stamp = stamp
        self.is_wholescene = is_wholescene
        self.scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_factor)
        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "point_acc": -float("inf"),
            "point_acc_per_class": -float("inf"),
            "voxel_acc": -float("inf"),
            "voxel_acc_per_class": -float("inf"),
            "point_miou": -float("inf"),
            "voxel_miou": -float("inf"),
        }

        # log
        # contains all necessary info for all phases
        self.log = {phase: {} for phase in ["train", "val"]}
        
        # tensorboard
        tb_path = os.path.join(CONF.OUTPUT_ROOT, stamp, "tensorboard")
        os.makedirs(tb_path, exist_ok=True)
        self._log_writer = SummaryWriter(tb_path)

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * epoch
        
        for epoch_id in range(epoch):
            print("epoch {} starting...".format(epoch_id + 1))
            # train
            self._set_phase("train")
            self._train(self.dataloader["train"], epoch_id)

            # val
            self._set_phase("eval")
            self._val(self.dataloader["val"], epoch_id)

            # epoch report
            self._epoch_report(epoch_id)

            # load tensorboard
            self._dump_log(epoch_id)

            # scheduler
            self.scheduler.step()

        # print best
        self._best_report()

        # save model
        print("saving last models...\n")
        model_root = os.path.join(CONF.OUTPUT_ROOT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        self._log_writer.export_scalars_to_json(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "tensorboard", "all_scalars.json"))

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "eval":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, coord, feat, is_wholescene):
        if self.is_wholescene:
            pred = []
            coord_chunk, feat_chunk = torch.split(coord.squeeze(0), self.batch_size, 0), torch.split(feat.squeeze(0), self.batch_size, 0)
            assert len(coord_chunk) == len(feat_chunk)
            for coord, feat in zip(coord_chunk, feat_chunk):
                output = self.model(torch.cat([coord, feat], dim=2))
                pred.append(output)

            pred = torch.cat(pred, dim=0).unsqueeze(0)
        else:
            output = self.model(torch.cat([coord, feat], dim=2))
            pred = output

        return pred

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        # self._clip_grad()
        self.optimizer.step()

    def _compute_loss(self, pred, target, weights):
        num_classes = pred.size(-1)
        loss = self.criterion(pred.contiguous().view(-1, num_classes), target.view(-1), weights.view(-1))
        self._running_log["loss"] = loss

    def _train(self, train_loader, epoch_id):
        # setting
        phase = "train"
        self.log[phase][epoch_id] = {
            # info
            "forward": [],
            "backward": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "point_acc": [],
            "point_acc_per_class": [],
            "voxel_acc": [],
            "voxel_acc_per_class": [],
            "point_miou": [],
            "voxel_miou": [],
        }
        for iter_id, data in enumerate(train_loader):
            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                # acc
                "point_acc": 0,
                "point_acc_per_class": 0,
                "voxel_acc": 0,
                "voxel_acc_per_class": 0,
                "point_miou": 0,
                "voxel_miou": 0,
            }

            # unpack the data
            coords, feats, semantic_segs, sample_weights, fetch_time = data
            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            self.log[phase][epoch_id]["fetch"].append(fetch_time)

            # forward
            start_forward = time.time()
            preds = self._forward(coords, feats, self.is_wholescene)
            self._compute_loss(preds, semantic_segs, sample_weights)
            self._eval(coords, preds, semantic_segs, sample_weights, self.is_wholescene)
            self.log[phase][epoch_id]["forward"].append(time.time() - start_forward)

            # backward
            start = time.time()
            self._backward()
            self.log[phase][epoch_id]["backward"].append(time.time() - start)

            # record log
            self.log[phase][epoch_id]["loss"].append(self._running_log["loss"].item())
            self.log[phase][epoch_id]["point_acc"].append(self._running_log["point_acc"])
            self.log[phase][epoch_id]["point_acc_per_class"].append(self._running_log["point_acc_per_class"])
            self.log[phase][epoch_id]["voxel_acc"].append(self._running_log["voxel_acc"])
            self.log[phase][epoch_id]["voxel_acc_per_class"].append(self._running_log["voxel_acc_per_class"])
            self.log[phase][epoch_id]["point_miou"].append(self._running_log["point_miou"])
            self.log[phase][epoch_id]["voxel_miou"].append(self._running_log["voxel_miou"])

            # report
            iter_time = self.log[phase][epoch_id]["fetch"][-1]
            iter_time += self.log[phase][epoch_id]["forward"][-1]
            iter_time += self.log[phase][epoch_id]["backward"][-1]
            self.log[phase][epoch_id]["iter_time"].append(iter_time)
            if (iter_id + 1) % self.verbose == 0:
                self._train_report(epoch_id)


            # update the _global_iter_id
            self._global_iter_id += 1

    def _val(self, val_loader, epoch_id):
        # setting
        phase = "val"
        self.log[phase][epoch_id] = {
            # info
            "forward": [],
            "backward": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "point_acc": [],
            "point_acc_per_class": [],
            "voxel_acc": [],
            "voxel_acc_per_class": [],
            "point_miou": [],
            "voxel_miou": [],
        }
        for iter_id, data in enumerate(val_loader):
            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                # acc
                "point_acc": 0,
                "point_acc_per_class": 0,
                "voxel_acc": 0,
                "voxel_acc_per_class": 0,
                "point_miou": 0,
                "voxel_miou": 0,
            }

            # unpack the data
            coords, feats, semantic_segs, sample_weights, fetch_time = data
            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            self.log[phase][epoch_id]["fetch"].append(fetch_time)

            # forward
            preds = self._forward(coords, feats, self.is_wholescene)
            self._compute_loss(preds, semantic_segs, sample_weights)
            self._eval(coords, preds, semantic_segs, sample_weights, self.is_wholescene)

            # record log
            self.log[phase][epoch_id]["loss"].append(self._running_log["loss"].item())
            self.log[phase][epoch_id]["point_acc"].append(self._running_log["point_acc"])
            self.log[phase][epoch_id]["point_acc_per_class"].append(self._running_log["point_acc_per_class"])
            self.log[phase][epoch_id]["voxel_acc"].append(self._running_log["voxel_acc"])
            self.log[phase][epoch_id]["voxel_acc_per_class"].append(self._running_log["voxel_acc_per_class"])
            self.log[phase][epoch_id]["point_miou"].append(self._running_log["point_miou"])
            self.log[phase][epoch_id]["voxel_miou"].append(self._running_log["voxel_miou"])

        # check best
        cur_criterion = "voxel_miou"
        cur_best = np.mean(self.log[phase][epoch_id][cur_criterion])
        if cur_best > self.best[cur_criterion]:
            print("best {} achieved: {}".format(cur_criterion, cur_best))
            print("current train_loss: {}".format(np.mean(self.log["train"][epoch_id]["loss"])))
            print("current val_loss: {}".format(np.mean(self.log["val"][epoch_id]["loss"])))
            self.best["epoch"] = epoch_id + 1
            self.best["loss"] = np.mean(self.log[phase][epoch_id]["loss"])
            self.best["point_acc"] = np.mean(self.log[phase][epoch_id]["point_acc"])
            self.best["point_acc_per_class"] = np.mean(self.log[phase][epoch_id]["point_acc_per_class"])
            self.best["voxel_acc"] = np.mean(self.log[phase][epoch_id]["voxel_acc"])
            self.best["voxel_acc_per_class"] = np.mean(self.log[phase][epoch_id]["voxel_acc_per_class"])
            self.best["point_miou"] = np.mean(self.log[phase][epoch_id]["point_miou"])
            self.best["voxel_miou"] = np.mean(self.log[phase][epoch_id]["voxel_miou"])

            # save model
            print("saving models...\n")
            model_root = os.path.join(CONF.OUTPUT_ROOT, self.stamp)
            torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _eval(self, coords, preds, targets, weights, is_wholescene):
        if is_wholescene:
            coords = coords.squeeze(0).view(-1, 3).cpu().numpy()            # (CK * N, 3)
            preds = preds.max(3)[1].squeeze(0).view(-1).cpu().numpy()       # (CK * N)
            targets = targets.squeeze(0).view(-1).cpu().numpy()             # (CK * N)
            weights = weights.squeeze(0).view(-1).cpu().numpy()             # (CK * N)
        else:
            coords = coords.view(-1, 3).cpu().numpy()            # (B * N, 3)
            preds = preds.max(2)[1].view(-1).cpu().numpy()       # (B * N)
            targets = targets.view(-1).cpu().numpy()             # (B * N)
            weights = weights.view(-1).cpu().numpy()             # (B * N)

        pointacc, pointacc_per_class, voxacc, voxacc_per_class, _, acc_mask = compute_acc(coords, preds, targets, weights)
        pointmiou, voxmiou, miou_mask = compute_miou(coords, preds, targets, weights)
        
        self._running_log["point_acc"] = pointacc
        self._running_log["point_acc_per_class"] = np.sum(pointacc_per_class * acc_mask)/np.sum(acc_mask)
        self._running_log["voxel_acc"] = voxacc
        self._running_log["voxel_acc_per_class"] = np.sum(voxacc_per_class * acc_mask)/np.sum(acc_mask)
        self._running_log["point_miou"] = np.sum(pointmiou * miou_mask)/np.sum(miou_mask)
        self._running_log["voxel_miou"] = np.sum(voxmiou * miou_mask)/np.sum(miou_mask)

    def _dump_log(self, epoch_id):
        # loss
        self._log_writer.add_scalars(
            "log/{}".format("loss"),
            {
                "train": np.mean([loss for loss in self.log["train"][epoch_id]["loss"]]),
                "val": np.mean([loss for loss in self.log["val"][epoch_id]["loss"]])
            },
            epoch_id
        )

        # eval
        self._log_writer.add_scalars(
            "eval/{}".format("point_acc"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["point_acc"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["point_acc"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("point_acc_per_class"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["point_acc_per_class"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["point_acc_per_class"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("voxel_acc"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["voxel_acc"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("voxel_acc_per_class"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc_per_class"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["voxel_acc_per_class"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("point_miou"),
            {
                "train": np.mean([miou for miou in self.log["train"][epoch_id]["point_miou"]]),
                "val": np.mean([miou for miou in self.log["val"][epoch_id]["point_miou"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("voxel_miou"),
            {
                "train": np.mean([miou for miou in self.log["train"][epoch_id]["voxel_miou"]]),
                "val": np.mean([miou for miou in self.log["val"][epoch_id]["voxel_miou"]])
            },
            epoch_id
        )

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = [time for time in self.log["train"][epoch_id]["fetch"]]
        forward_time = [time for time in self.log["train"][epoch_id]["forward"]]
        backward_time = [time for time in self.log["train"][epoch_id]["backward"]]
        iter_time = [time for time in self.log["train"][epoch_id]["iter_time"]]
        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * (self.epoch - epoch_id) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            global_iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([loss for loss in self.log["train"][epoch_id]["loss"]]), 5),
            train_point_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["point_acc"]]), 5),
            train_point_acc_per_class=round(np.mean([acc for acc in self.log["train"][epoch_id]["point_acc_per_class"]]), 5),
            train_voxel_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc"]]), 5),
            train_voxel_acc_per_class=round(np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc_per_class"]]), 5),
            train_point_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["point_miou"]]), 5),
            train_voxel_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["voxel_miou"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        print(iter_report)

    def _epoch_report(self, epoch_id):
        print("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([loss for loss in self.log["train"][epoch_id]["loss"]]), 5),
            train_point_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["point_acc"]]), 5),
            train_point_acc_per_class=round(np.mean([acc for acc in self.log["train"][epoch_id]["point_acc_per_class"]]), 5),
            train_voxel_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc"]]), 5),
            train_voxel_acc_per_class=round(np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc_per_class"]]), 5),
            train_point_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["point_miou"]]), 5),
            train_voxel_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["voxel_miou"]]), 5),
            val_loss=round(np.mean([loss for loss in self.log["val"][epoch_id]["loss"]]), 5),
            val_point_acc=round(np.mean([acc for acc in self.log["val"][epoch_id]["point_acc"]]), 5),
            val_point_acc_per_class=round(np.mean([acc for acc in self.log["val"][epoch_id]["point_acc_per_class"]]), 5),
            val_voxel_acc=round(np.mean([acc for acc in self.log["val"][epoch_id]["voxel_acc"]]), 5),
            val_voxel_acc_per_class=round(np.mean([acc for acc in self.log["val"][epoch_id]["voxel_acc_per_class"]]), 5),
            val_point_miou=round(np.mean([miou for miou in self.log["val"][epoch_id]["point_miou"]]), 5),
            val_voxel_miou=round(np.mean([miou for miou in self.log["val"][epoch_id]["voxel_miou"]]), 5),
        )
        print(epoch_report)
    
    def _best_report(self):
        print("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            point_acc=round(self.best["point_acc"], 5),
            point_acc_per_class=round(self.best["point_acc_per_class"], 5),
            voxel_acc=round(self.best["voxel_acc"], 5),
            voxel_acc_per_class=round(self.best["voxel_acc_per_class"], 5),
            point_miou=round(self.best["point_miou"], 5),
            voxel_miou=round(self.best["voxel_miou"], 5),
        )
        print(best_report)
        with open(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
