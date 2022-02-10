#!/usr/bin/env python3
#
# Decoupled yolox detection head
# this class is derived from models.YOLOXHead
# modified by Pavel Bystrov

import torch
import math
from torch.nn import BCEWithLogitsLoss, Conv2d, L1Loss, ModuleList, Sequential

from .yolo_head import YOLOXHead
from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class DecoupledYOLOHead(YOLOXHead):
    """
      Decoupled yolox head class derived from YOLOXHead
      DecoupledYOLOHead parameter reg_convs is not used anymore
    """
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Arguments:
            num_classes : number of classes
            width: width of bounding box
            strides: array with strides for different FPN layers
            in_channels: array with input channels for RPN 
            act: activation type for convolution. Defalut value: "silu"
            depthwise: whether apply depthwise conv in conv branch. Defalut value: False
        """
        super().__init__(num_classes, width=width, strides=strides, in_channels=in_channels, act=act, depthwise=depthwise) ##super(DecoupledYOLOHead, self)
        Conv = DWConv if depthwise else BaseConv
        self.reg_convs = ModuleList()
        self.obj_convs = ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1, act=act)
            )
            self.cls_convs.append(
                Sequential(
                        Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                        Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                )
            )
            self.cls_preds.append(
                Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * self.num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_convs.append(
                Sequential(
                        Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                        Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                )
            )
            self.obj_convs.append(
                Sequential(
                        Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                        Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                )
            )
            self.reg_preds.append(
                Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 1, kernel_size=1, stride=1, padding=0)
            )
        self.l1_loss = L1Loss(reduction="none")
        self.bcewithlog_loss = BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        print(self.obj_convs)
        print(self.cls_convs)


    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def forward(self, xin, labels=None, imgs=None):
        """
        Arguments:
            xin : inputs
            labels: class labels
            imgs: images
        """
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](reg_x)
            obj_feet = self.obj_convs[k](reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](obj_feet)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs] # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs


    def freeze_layers(self, layer1, layer2):
        for param in layer1.parameters():
            param.requires_grad = False
        for param in layer2.parameters():
            param.requires_grad = False


    def freeze_3_layers(self, layer1, layer2, layer3):
        for param in layer1.parameters():
            param.requires_grad = False
        for param in layer2.parameters():
            param.requires_grad = False
        for param in layer3.parameters():
            param.requires_grad = False


    def unfreeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True


    def unfreeze_layers(self, layer1, layer2):
        for param in layer1.parameters():
            param.requires_grad = True
        for param in layer2.parameters():
            param.requires_grad = True


    def unfreeze_3_layers(self, layer1, layer2, layer3):
        for param in layer1.parameters():
            param.requires_grad = True
        for param in layer2.parameters():
            param.requires_grad = True
        for param in layer3.parameters():
            param.requires_grad = True


class ParallelYOLOHead(YOLOXHead):
    """
      Decoupled yolox head class derived from YOLOXHead
      DecoupledYOLOHead parameter reg_convs is not used anymore
    """
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Arguments:
            num_classes : number of classes
            width: width of bounding box
            strides: array with strides for different FPN layers
            in_channels: array with input channels for RPN 
            act: activation type for convolution. Defalut value: "silu"
            depthwise: whether apply depthwise conv in conv branch. Defalut value: False
        """
        super().__init__(num_classes, width=width, strides=strides, in_channels=in_channels, act=act, depthwise=depthwise) ##super(DecoupledYOLOHead, self)
        Conv = DWConv if depthwise else BaseConv
        self.reg_convs = ModuleList()
        self.obj_convs = ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1, act=act)
            )
            self.cls_convs.append(
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            )
            self.cls_preds.append(
                Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * self.num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_convs.append(
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            )
            self.obj_convs.append(
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            )
            self.reg_preds.append(
                Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 1, kernel_size=1, stride=1, padding=0)
            )
        self.l1_loss = L1Loss(reduction="none")
        self.bcewithlog_loss = BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        print(self.obj_convs)
        print(self.cls_convs)


    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def forward(self, xin, labels=None, imgs=None):
        """
        Arguments:
            xin : inputs
            labels: class labels
            imgs: images
        """
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](reg_x)
            obj_feet = self.obj_convs[k](reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](obj_feet)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs] # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs


    def freeze_layers(self, layer1, layer2):
        for param in layer1.parameters():
            param.requires_grad = False
        for param in layer2.parameters():
            param.requires_grad = False


    def freeze_3_layers(self, layer1, layer2, layer3):
        for param in layer1.parameters():
            param.requires_grad = False
        for param in layer2.parameters():
            param.requires_grad = False
        for param in layer3.parameters():
            param.requires_grad = False


    def unfreeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True


    def unfreeze_layers(self, layer1, layer2):
        for param in layer1.parameters():
            param.requires_grad = True
        for param in layer2.parameters():
            param.requires_grad = True


    def unfreeze_3_layers(self, layer1, layer2, layer3):
        for param in layer1.parameters():
            param.requires_grad = True
        for param in layer2.parameters():
            param.requires_grad = True
        for param in layer3.parameters():
            param.requires_grad = True

