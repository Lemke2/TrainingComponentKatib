import torch.utils.data.dataloader
import random
import os
from data_utils import *
from loss_utils import *
from model_utils import *
from metrics_utils import*
from configUnet3 import config_func_unet3
import argparse
import logging


def end_of_epoch_print(epoch,all_validation_losses):
    ispis = "{{metricName: loss, metricValue: {:.4f}}}\n".format(all_validation_losses[epoch])
    print(ispis)
    logging.info(ispis)

def main(putanja_train, putanja_val, putanja_test, p_index,lr,lambda_p,step, num_epochs, loss_type,Batch_size, cfg):

    lr = lr
    lambda_parametri = lambda_p
    stepovi = step
    epochs = num_epochs
    batch_size = Batch_size

    ####################
    ### data loading ###
    ####################

    train_loader, valid_loader = data_loading(putanja_train, putanja_val, cfg.binary, p_index, cfg.net_type, batch_size)

    ############################
    ### model initialization ###
    ############################

    segmentation_net = model_init(cfg.num_channels, cfg.num_channels_lab, cfg.img_h, cfg.img_w, cfg.zscore,
                                cfg.net_type, cfg.device, cfg.server, cfg.GPU_list)
    segmentation_net = torch.nn.DataParallel(segmentation_net, device_ids=[0]).to(cfg.device)

    ############################
    ### model initialization ###
    ############################


    optimizer, scheduler = optimizer_init(segmentation_net, lr, cfg.weight_decay, cfg.scheduler_lr, lambda_parametri, cfg.optimizer_patience)

    ############################
    ### Loss initialization ###
    ############################

    criterion = loss_init(cfg.use_weights, loss_type, cfg.dataset, cfg.num_channels_lab, cfg.device, cfg.use_mask)

    # Brojanje Iteracija
    epoch_list = np.zeros([epochs])
    all_train_losses = np.zeros([epochs])
    all_validation_losses = np.zeros([epochs])

    for epoch in range(epochs):

        train_part = "Train"
        segmentation_net.train(mode=True)

        index_start = 0

        batch_iou = torch.zeros(size=(len(train_loader.dataset.img_names), cfg.num_channels_lab*2), device= cfg.device, dtype=torch.float32)

        loss_type = 'bce'

        if loss_type == 'bce':
            batch_iou_bg = torch.zeros(size=(len(train_loader.dataset.img_names),2), device = cfg.device, dtype=torch.float32)


        for input_var, target_var, batch_names_train in train_loader:

            set_zero_grad(segmentation_net)

            model_output = segmentation_net.forward(input_var)
            loss = loss_calc(loss_type, criterion, model_output, target_var, cfg.num_channels_lab, cfg.use_mask)
            loss.backward()

            optimizer.step()  # mnozi sa grad i menja weightove

            cfg.train_losses.append(loss.data)

            ######## update!!!!

            index_end = index_start + len(batch_names_train)
            if loss_type == 'bce':
                batch_iou[index_start:index_end, :], batch_iou_bg[index_start:index_end] = calc_metrics_pix(model_output, target_var,
                 cfg.num_channels_lab, cfg.device, cfg.use_mask, loss_type)
            elif loss_type == 'ce':
                batch_iou[index_start:index_end, :]= calc_metrics_pix(model_output, target_var, 
                                                        cfg.mask_train, cfg.num_channels_lab, cfg.device, cfg.use_mask,loss_type)
            else:
                print("Error: unimplemented loss type")
                sys.exit(0)
            index_start += len(batch_names_train)

            cfg.count_train += 1

        if loss_type == 'bce':
            final_metric_calculation(loss_type = loss_type,epoch=epoch,num_channels_lab = cfg.num_channels_lab,classes_labels = cfg.classes_labels,
            batch_iou_bg=batch_iou_bg,batch_iou=batch_iou,train_part= train_part)
        elif loss_type == 'ce':
            final_metric_calculation(loss_type = loss_type,epoch=epoch,num_channels_lab=cfg.num_channels_lab,classes_labels = cfg.classes_labels,
            batch_iou=batch_iou,train_part= train_part)
        else:
            print("Error: Unimplemented loss type!")
            sys.exit(0)

        all_train_losses[epoch] = (torch.mean(torch.tensor(cfg.train_losses,dtype = torch.float32)))


        if epoch !=0 and (epoch % stepovi)==0 :
            print("epoha: " + str(epoch) +" , uradjen step!")

        if cfg.server:
            torch.cuda.empty_cache()
        train_part = "Valid"
        segmentation_net.eval()

        index_start = 0

        batch_iou = torch.zeros(size=(len(valid_loader.dataset.img_names), cfg.num_channels_lab*2),device=cfg.device,dtype=torch.float32)
        if loss_type == 'bce':
            batch_iou_bg = torch.zeros(size=(len(valid_loader.dataset.img_names),2),device=cfg.device,dtype=torch.float32)

        for input_var, target_var, batch_names_valid in valid_loader:

            model_output = segmentation_net.forward(input_var)
            val_loss = loss_calc(loss_type,criterion,model_output,target_var, cfg.num_channels_lab ,cfg.use_mask)

            cfg.validation_losses.append(val_loss.data)

            index_end = index_start + len(batch_names_valid)
            if loss_type == 'bce':
                batch_iou[index_start:index_end, :], batch_iou_bg[index_start:index_end]= calc_metrics_pix(model_output, target_var,cfg.num_channels_lab,cfg.device,cfg.use_mask,loss_type)
            elif loss_type == 'ce':
                batch_iou[index_start:index_end, :]= calc_metrics_pix(model_output, target_var, cfg.num_channels_lab, cfg.device, cfg.use_mask,loss_type)
            else:
                print("Error: unimplemented loss type")
                sys.exit(0)

            index_start += len(batch_names_valid)

            cfg.count_val += 1

        scheduler.step(torch.mean(torch.tensor(cfg.validation_losses)))
        ##############################################################
        ### Racunanje finalne metrike nad celim validacionim setom ###
        ##############################################################

        if loss_type == 'bce':
            final_metric_calculation(loss_type = loss_type, epoch=epoch,num_channels_lab = cfg.num_channels_lab,
            classes_labels = cfg.classes_labels, batch_iou_bg = batch_iou_bg, batch_iou = batch_iou, train_part = train_part)
        elif loss_type == 'ce':
            final_metric_calculation(loss_type = loss_type, epoch = epoch, num_channels_lab = cfg.num_channels_lab, 
            classes_labels = cfg.classes_labels, batch_iou = batch_iou, train_part = train_part)
        else:
            print("Error: Unimplemented loss type!")
            sys.exit(0)

        epoch_list[epoch] = epoch
        all_validation_losses[epoch] = (torch.mean(torch.tensor(cfg.validation_losses,dtype = torch.float32)))

        end_of_epoch_print(epoch,all_validation_losses)

if __name__ == '__main__':
    config = config_func_unet3(False)
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar="N")
    parser.add_argument('--lambda_parametar', type=int, default=0.99, metavar="N")
    parser.add_argument('--stepovi_arr', type=int, default=5, metavar="N")
    parser.add_argument('--num_epochs', type=int, default=3, metavar="N")
    parser.add_argument('--loss_type', type=str, default="bce", metavar="N")
    parser.add_argument('--Batch_size', type=int, default=8, metavar="N")
    parser.add_argument('--trening_location', type=str)
    parser.add_argument('--validation_location', type=str)
    parser.add_argument('--test_location', type=str)
    parser.add_argument('--new_location', type=str)
    args = parser.parse_args()
    
    data_location = args.new_location

    logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
            filename=data_location)

    learning_rate = args.learning_rate
    lambda_parametar = args.lambda_parametar
    stepovi_arr = args.stepovi_arr

    num_epochs = args.num_epochs
    loss_type = args.loss_type
    Batch_size = args.Batch_size

    trening_location = args.trening_location
    validation_location = args.validation_location
    test_location = args.test_location

    print("---inputs----")
    print(learning_rate)
    print(lambda_parametar)
    print(stepovi_arr)
    print(num_epochs)
    print(loss_type)
    print(Batch_size)
    print(trening_location)
    print(validation_location)
    print(test_location)
    print("-----------------")
    param_ponovljivosti = 1

    main(trening_location, validation_location, test_location, param_ponovljivosti, learning_rate, lambda_parametar, stepovi_arr,
                 num_epochs, loss_type, Batch_size, config)
    print("End of training component")
