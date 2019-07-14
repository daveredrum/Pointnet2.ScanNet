import os
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter

import sys
sys.path.append(".")
from lib.utils import decode_eta
from lib.config import CONF

ITER_REPORT_TEMPLATE = """
----------------------iter: [{global_iter_id}/{total_iter}]----------------------
[loss] train_loss: {train_loss}
[sco.] train_acc: {train_acc}
[sco.] train_miou: {train_miou}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
------------------------summary------------------------
[train] train_loss: {train_loss}
[train] train_acc: {train_acc}
[train] train_miou: {train_miou}
[val]   val_loss: {val_loss}
[val]   val_acc: {val_acc}
[val]   val_miou: {val_miou}
"""

BEST_REPORT_TEMPLATE = """
-----------------------------best-----------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[sco.] acc: {acc}
[sco.] miou: {miou}
"""

class Solver():
    def __init__(self, model, dataloader, criterion, optimizer, batch_size, stamp, is_wholescene=True):
        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.stamp = stamp
        self.is_wholescene = is_wholescene
        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "acc": -float("inf"),
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

    def _forward(self, coord, feat):
        if self.is_wholescene:
            pred = []
            coord_chunk, feat_chunk = torch.split(coord.squeeze(0), self.batch_size, 0), torch.split(feat.squeeze(0), self.batch_size, 0)
            assert len(coord_chunk) == len(feat_chunk)
            for coord, feat in zip(coord_chunk, feat_chunk):
                output = self.model(torch.cat([coord, feat], dim=2))
                pred.append(output)

            pred = torch.cat(pred, dim=0)
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
        num_classes = pred.size(2)
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
            # constraint loss (float, not torch.cuda.FloatTensor)
            "acc": [],
            "miou": []
        }
        for iter_id, data in enumerate(train_loader):
            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                # acc
                "acc": 0,
                "miou": 0
            }

            # unpack the data
            coords, feats, semantic_segs, sample_weights, fetch_time = data
            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            self.log[phase][epoch_id]["fetch"].append(fetch_time)

            # forward
            start_forward = time.time()
            preds = self._forward(coords, feats)
            self._compute_loss(preds, semantic_segs, sample_weights)
            self._eval(preds, semantic_segs)
            self.log[phase][epoch_id]["forward"].append(time.time() - start_forward)

            # backward
            start = time.time()
            self._backward()
            self.log[phase][epoch_id]["backward"].append(time.time() - start)

            # record log
            self.log[phase][epoch_id]["loss"].append(self._running_log["loss"].item())
            self.log[phase][epoch_id]["acc"].append(self._running_log["acc"])
            self.log[phase][epoch_id]["miou"].append(self._running_log["miou"])

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
            # constraint loss (float, not torch.cuda.FloatTensor)
            "acc": [],
            "miou": []
        }
        for iter_id, data in enumerate(val_loader):
            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                # acc
                "acc": 0,
                "miou": 0
            }

            # unpack the data
            coords, feats, semantic_segs, sample_weights, fetch_time = data
            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            self.log[phase][epoch_id]["fetch"].append(fetch_time)

            # forward
            preds = self._forward(coords, feats)
            self._compute_loss(preds, semantic_segs, sample_weights)
            self._eval(preds, semantic_segs)

            # record log
            self.log[phase][epoch_id]["loss"].append(self._running_log["loss"].item())
            self.log[phase][epoch_id]["acc"].append(self._running_log["acc"])
            self.log[phase][epoch_id]["miou"].append(self._running_log["miou"])

        # check best
        cur_criterion = "acc"
        cur_best = np.mean(self.log[phase][epoch_id][cur_criterion])
        if cur_best > self.best[cur_criterion]:
            print("best {} achieved: {}".format(cur_criterion, cur_best))
            print("current train_loss: {}".format(np.mean(self.log["train"][epoch_id]["loss"])))
            print("current val_loss: {}".format(np.mean(self.log["val"][epoch_id]["loss"])))
            self.best["epoch"] = epoch_id + 1
            self.best["loss"] = np.mean(self.log[phase][epoch_id]["loss"])
            self.best["acc"] = np.mean(self.log[phase][epoch_id]["acc"])
            self.best["miou"] = np.mean(self.log[phase][epoch_id]["miou"])

            # save model
            print("saving models...\n")
            model_root = os.path.join(CONF.OUTPUT_ROOT, self.stamp)
            torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _eval(self, preds, targets):
        if self.is_wholescene:
            preds = preds.max(2)[1]
            targets = targets.squeeze(0)
        else:
            preds = preds.max(2)[1]

        # num_correct_nonzero = preds.eq(targets).sum().item() - preds[preds == 0].view(-1).size(0)
        # num_total_nonzero = preds.view(-1).size(0) - preds[preds == 0].view(-1).size(0)
        # self._running_log["acc"] = num_correct_nonzero / num_total_nonzero
        self._running_log["acc"] = preds.eq(targets).sum().item() / preds.view(-1).size(0)

        miou = []
        for i in range(21):
            # if i == 0: continue
            pred_ids = torch.arange(preds.view(-1).size(0))[preds.view(-1) == i].tolist()
            target_ids = torch.arange(targets.view(-1).size(0))[targets.view(-1) == i].tolist()
            if len(target_ids) == 0: continue
            num_correct = len(set(pred_ids).intersection(set(target_ids)))
            num_union = len(set(pred_ids).union(set(target_ids)))
            miou.append(num_correct / (num_union + 1e-8))

        self._running_log["miou"] = np.mean(miou)

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
            "eval/{}".format("acc"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["acc"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["acc"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("miou"),
            {
                "train": np.mean([miou for miou in self.log["train"][epoch_id]["miou"]]),
                "val": np.mean([miou for miou in self.log["val"][epoch_id]["miou"]])
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
            train_acc=round(np.mean([loss for loss in self.log["train"][epoch_id]["acc"]]), 5),
            train_miou=round(np.mean([loss for loss in self.log["train"][epoch_id]["miou"]]), 5),
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
            train_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["acc"]]), 5),
            train_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["miou"]]), 5),
            val_loss=round(np.mean([loss for loss in self.log["val"][epoch_id]["loss"]]), 5),
            val_acc=round(np.mean([acc for acc in self.log["val"][epoch_id]["acc"]]), 5),
            val_miou=round(np.mean([miou for miou in self.log["val"][epoch_id]["miou"]]), 5),
        )
        print(epoch_report)
    
    def _best_report(self):
        print("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            acc=round(self.best["acc"], 5),
            miou=round(self.best["miou"], 5),
        )
        print(best_report)
        with open(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
