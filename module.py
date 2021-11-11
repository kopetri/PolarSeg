import torch
import pytorch_lightning as pl
import numpy as np
from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from network.lovasz_losses import lovasz_softmax

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count=np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label)+1)
    hist=hist[unique_label,:]
    hist=hist[:,unique_label]
    return hist

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick


class PolarNetModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        if self.opt.model == 'polar':
            fea_dim = 9
            circular_padding = True
        elif self.opt.model == 'traditional':
            fea_dim = 7
            circular_padding = False
        
        #prepare miou fun
        self.opt.unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        unique_label_str=[SemKITTI_label_name[x] for x in self.opt.unique_label+1]
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        #prepare model
        self.my_BEV_model=BEV_Unet(n_class=len(self.opt.unique_label), 
                                   n_height=self.opt.grid_size[2],
                                   input_batch_norm=True,
                                   dropout=0.5,
                                   circular_padding=circular_padding)

        self.my_model = ptBEVnet(self.my_BEV_model, 
                                pt_model='pointnet',
                                grid_size=self.opt.grid_size,
                                fea_dim=fea_dim,
                                max_pt_per_encode=256,
                                out_pt_fea_dim=512,
                                kernal_size=1,
                                pt_selection='random',
                                fea_compre=self.opt.grid_size[2])
    def forward(self, feat_ten, grid):
        pred = self.my_model(feat_ten, grid) 
        return pred
        
    def training_step(self, batch, batch_idx):
        _,train_vox_label,train_grid,_,train_pt_fea = batch
        train_vox_label = SemKITTI2train(train_vox_label)
        train_pt_fea_ten = [torch.from_numpy(i).to(self.device).float() for i in train_pt_fea]
        train_grid_ten = [torch.from_numpy(i[:,:2]).to(self.device).float() for i in train_grid]
        train_vox_ten = [torch.from_numpy(i).to(self.device).float() for i in train_grid]
        point_label_tensor=train_vox_label.type(torch.LongTensor).to(self.device)

        # forward + backward + optimize
        outputs = self(train_pt_fea_ten,train_grid_ten)
        loss = lovasz_softmax(torch.nn.functional.softmax(outputs, dim=1), point_label_tensor,ignore=255) + self.criterion(outputs,point_label_tensor)
        self.log('train_log', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _,val_vox_label,val_grid,val_pt_labs,val_pt_fea = batch

        val_vox_label = SemKITTI2train(val_vox_label)
        val_pt_labs = SemKITTI2train(val_pt_labs)
        val_pt_fea_ten = [torch.from_numpy(i).to(self.device).float() for i in val_pt_fea]
        val_grid_ten = [torch.from_numpy(i[:,:2]).to(self.device).float() for i in val_grid]
        val_label_tensor=val_vox_label.type(torch.LongTensor).to(self.device)

        predict_labels = self(val_pt_fea_ten, val_grid_ten)
        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels, dim=1).detach(), val_label_tensor,ignore=255) + self.criterion(predict_labels.detach(),val_label_tensor)
        predict_labels = torch.argmax(predict_labels,dim=1)
        predict_labels = predict_labels.cpu().detach().numpy()
        hist_list = []
        for count,i_val_grid in enumerate(val_grid):
            hist_list.append(fast_hist_crop(predict_labels[count,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]],val_pt_labs[count],self.opt.unique_label))

        iou = per_class_iu(sum(hist_list))
        val_miou = np.nanmean(iou)

        self.log("valid_loss", loss)
        self.log("valid_miou", val_miou)
        return {'valid_loss': loss, 'valid_miou': val_miou}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.my_model.parameters())

        return optimizer
