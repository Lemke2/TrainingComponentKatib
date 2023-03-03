import numpy as np
import torch
import sys
from configUnet3 import config_func_unet3
cfg = config_func_unet3(False)

def final_metric_calculation(loss_type = 'bce',epoch=0,num_channels_lab = 1,classes_labels ='None',batch_iou_bg='None',batch_iou='None',train_part= 'Test'):
    index_miou = 0  
    IOU = list() 
    if loss_type == 'bce':
        iou_int_bg = batch_iou_bg[:,0]
        iou_un_bg = batch_iou_bg[:,1]
        iou_calc_bg = torch.div(torch.sum(iou_int_bg),torch.sum(iou_un_bg))
        if train_part == 'Valid':
            ispis = train_part + " Mean IOU Classwise/" + "Background" + " "+str(np.round(iou_calc_bg.detach().cpu(), 4))
            IOU.append(np.round(iou_calc_bg.detach().cpu().numpy(), 4))
            print(ispis)
        else:
            ispis = train_part + " Mean IOU Classwise/" + "Background" + " "
    
    for klasa in range(num_channels_lab):
        
        iou_int = batch_iou[:, index_miou]
        iou_un = batch_iou[:, index_miou + 1]
        iou_calc = torch.div(torch.sum(iou_int),torch.sum(iou_un))
        index_miou += 2
        if train_part == 'Valid':
            ispis = train_part+ " Mean IOU Classwise/" + classes_labels[klasa] + " " +str(np.round(iou_calc.detach().cpu(), 4))
            IOU.append(np.round(iou_calc.detach().cpu().numpy(), 4))
            print(ispis)
        else:
            ispis = train_part+ " Mean IOU Classwise/" + classes_labels[klasa] + " "
        
    return IOU
    


def iou_pix(target, pred, use_mask):
    
    if torch.sum(target) == 0 and torch.sum(pred) == 0:
        arr = torch.full(size=(target.shape[0], target.shape[1]),fill_value=2)
        return arr[arr!=2].sum(), arr[arr!=2].sum()

    else:

        if use_mask:
            intersection = torch.logical_and(target.bool(),pred.bool())[cfg.mask_var].sum()
            union = torch.logical_or(target.bool(), pred.bool())[cfg.mask_var].sum() 
            return intersection , union
        else:
            intersection = torch.logical_and(target.bool(), pred.bool()).sum()
            union = torch.logical_or(target.bool(), pred.bool()).sum() 
            return intersection , union 
        
def calc_metrics_pix(model_output, target_var, num_classes,device,use_mask,loss_type):
    sigmoid_func = torch.nn.Sigmoid()
    iou_res = torch.zeros((target_var.shape[0], target_var.shape[1] * 2),device=device)
    if loss_type == 'bce':
        iou_res_bg = torch.zeros((target_var.shape[0], 2),device=device)
        model_output = sigmoid_func(model_output)
    for im_number in range(target_var.shape[0]):
        
        tresholded = model_output[im_number, :, :, :]>0.5
        tresholded_tmp = tresholded.byte()
        
        if loss_type=='bce':
            bg_tresholded = (tresholded_tmp==0)
            bg_target_var = torch.max(target_var[im_number,:,:,:],dim = 0).values
            bg_target_var = (bg_target_var==0)
            iou_res_bg[ im_number, 0], iou_res_bg[ im_number, 1] = iou_pix( bg_target_var, bg_tresholded,use_mask)
        
        ind_iou = 0
        for klasa_idx in range(num_classes):

            iou_res[im_number, ind_iou], iou_res[im_number, ind_iou + 1] = iou_pix(target_var[im_number, klasa_idx, :, :], tresholded[klasa_idx,:,:],use_mask)
            ind_iou += 2
    if loss_type =='ce':       
        return iou_res
    elif loss_type == 'bce':
        return iou_res,iou_res_bg
    else:
        print("Error: Unimplemented loss type")
        sys.exit(0)


def calc_metrics_tb(model_output, target_var, num_classes,use_mask):
    miou_mean = []
    sigmoid_func = torch.nn.Sigmoid()
    model_output = sigmoid_func(model_output)
    for batch in range(target_var.shape[0]):    
        miou_res = torch.zeros([num_classes])
        tresholded = model_output[batch, :, :, :]>0.5
        tresholded = tresholded.byte()
        
        for klasa_idx in range(num_classes):

            miou_res[klasa_idx] = iou_coef(target_var.permute(0, 2, 3, 1)[batch, :, :, klasa_idx].byte(),tresholded[klasa_idx,:,:],use_mask)
 
        miou_res = [x for x in miou_res if torch.isnan(x) == False]
        
        miou_mean.append(torch.mean(torch.tensor(miou_res,dtype = torch.float32)))
        

    return miou_mean


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    smooth = 0
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def iou_coef(y_true,y_pred, use_mask):
    if use_mask:

        y_true_f = y_true[cfg.mask_var]
        y_pred_f = y_pred[cfg.mask_var]
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    smooth = 0
    return (intersection + smooth) / (union + smooth)