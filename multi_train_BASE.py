'''
The default exp_name is tmp. Change it before formal training!
multi-training is from https://github.com/liuquande/MS-Net/blob/master/train.py
nohup python -u multi_train_BASE.py --exp_name test --config_yml Configs/multi_train_local.yml --model SwinUnet --batch_size 4 --adapt_method False --dataset isic2018 PH2 DMF SKD --k_fold 4 > test.out 2>&1 &
'''
import argparse
from sqlite3 import adapt
import yaml
import os, time
from datetime import datetime

import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics
from torch.utils.tensorboard import SummaryWriter

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.losses import dice_loss
from Utils.pieces import DotDict


torch.cuda.empty_cache()

def main(config):
    # set gpu
    device_ids = range(torch.cuda.device_count())
    
    # prepare train, val, test datas
    train_loaders = {}  # initialize data loaders
    val_loaders = {}
    test_loaders = {}
    for dataset_name in config.data.name:
        datas = Dataset_wrap_csv(k_fold=config.data.k_fold, use_old_split=True, img_size=config.data.img_size, 
            dataset_name = dataset_name, split_ratio=config.data.split_ratio, 
            train_aug=config.data.train_aug, data_folder=config.data.data_folder)
        train_data, val_data, test_data = datas['train'], datas['test'], datas['test']

        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.train.batch_size,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
        train_loaders[dataset_name] = train_loader
        val_loaders[dataset_name] = val_loader
        test_loaders[dataset_name] = test_loader
        print('{} has {} training samples'.format(dataset_name, len(train_loader.dataset)))
    print('{} k_folder, {} val'.format(config.data.k_fold, config.data.use_val))

    
    # prepare model
    if config.model == 'BASE':
        from Models.Transformer.base import BASE
        model = BASE(drop_rate=0.1, drop_path_rate=0.1, conv_norm=nn.BatchNorm2d, adapt_method=config.model_adapt.adapt_method)
    elif config.model == 'BASE_DSN':
        from Models.Transformer.base import BASE_DSN
        model = BASE_DSN(img_size=config.data.img_size, drop_rate=0.1, drop_path_rate=0.1,
        conv_norm=nn.BatchNorm2d, adapt_method=config.model_adapt.adapt_method, num_domains=K)
    elif config.model == 'SwimUNETR':
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(img_size=(256,256), in_channels=3, out_channels=1, feature_size=48, use_checkpoint=False, spatial_dims=2)
    elif config.model == 'BASE_DASE':
        from Models.Sota_adapters.base_sota_adapt import BASE_DASE
        model = BASE_DASE(drop_rate=0.1, drop_path_rate=0.1, conv_norm=nn.BatchNorm2d)
    elif config.model == 'BASE_USE':
        from Models.Sota_adapters.base_sota_adapt import BASE_USE
        model =  BASE_USE(drop_rate=0.1, drop_path_rate=0.1, conv_norm=nn.BatchNorm2d)
    elif config.model == 'UTNet':
        from Models.Hybrid_models.UTNetFolder.UTNet import UTNet
        model = UTNet(in_chan=3,base_chan=32,num_classes=1,reduce_size=8,block_list='1234',num_blocks=[1,1,1,1],
        num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    elif config.model == 'SwinUnet':
        from Models.Transformer.SwinUnet import SwinUnet
        model  = SwinUnet(img_size=config.data.img_size)

    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))

    model = model.cuda()

    # If multiple GPUs
    if len(device_ids) > 1: 
        model = torch.nn.DataParallel(model).cuda()
    
    criterion = [nn.BCELoss(), dice_loss]

    # only test
    if config.test.only_test == True:
        test(config, model, config.test.test_model_dir, test_loaders, criterion)
    else:
        train_val(config, model, train_loaders, val_loaders, criterion)
        test(config, model, best_model_dir, test_loaders, criterion)



# =======================================================================================================
def train_val(config, model, train_loaders, val_loaders, criterion):
    # optimizer loss
    if config.train.optimizer.mode == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=float(config.train.optimizer.adamw.lr), 
        weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    max_iou = 0 # use for record best model
    best_epoch = 0 # use for recording the best epoch
    # create training data loading iteration
    train_iters = {}
    for dataset_name in train_loaders.keys():
        train_iters[dataset_name] = iter(train_loaders[dataset_name])
    if config.train.num_iters:
        iterations = config.train.num_iters
    else:
        iterations = max([len(train_loaders[x]) for x in train_iters.keys()])
    
    torch.save(model.state_dict(), best_model_dir)
    for epoch in range(epochs):
        start = time.time()
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model.train()
        for train_step in range(epoch*iterations, (epoch+1)*iterations):
            # for each dataset, get one minibatch, get loss, sum all losses together
            # update once
            datas_loss_list = []  #record loss for datasets
            dice_train_list = []
            iou_train_list = []
            for dataset_name in config.data.name:
                try:
                    batch = next(train_iters[dataset_name])
                except StopIteration:
                    train_iters[dataset_name] = iter(train_loaders[dataset_name])
                    batch = next(train_iters[dataset_name])
                img = batch['image'].cuda().float()
                label = batch['label'].cuda().float()
                domain_label = batch['set_id']
                d = str(domain_label[0].item())
                domain_label = torch.nn.functional.one_hot(domain_label, 4).float().cuda()
                if config.model == 'BASE_DSN' and config.model_adapt.adapt_method == 'Sup':
                    output = model(img,domain_label=domain_label,d=d)
                elif config.model_adapt.adapt_method and 'Sup' in config.model_adapt.adapt_method:
                    if config.model_adapt.Sup_label == 'Domain':
                        output = model(img, domain_label)
                elif config.model == 'BASE_DSN' or config.model_adapt.adapt_method in ['series_adapters', 'parallel_adapters']:
                    output = model(img,d=d)
                else:
                    output = model(img)
                output = torch.sigmoid(output)
    
                # calculate loss
                assert (output.shape == label.shape)
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss = sum(losses)
                datas_loss_list.append(loss)

                # calculate metrics
                with torch.no_grad():
                    output = output.cpu().numpy() > 0.5
                    label = label.cpu().numpy()
                    assert (output.shape == label.shape)
                    dice_train = metrics.dc(output, label)
                    iou_train = metrics.jc(output, label)
                    dice_train_list.append(dice_train)
                    iou_train_list.append(iou_train)

                # logging per batch
                writer.add_scalar('Train/{}/BCEloss'.format(dataset_name), losses[0].item(), train_step)
                writer.add_scalar('Train/{}/Diceloss'.format(dataset_name), losses[1].item(), train_step)
                writer.add_scalar('Train/{}/loss'.format(dataset_name), loss.item(), train_step)
                writer.add_scalar('Train/{}/Di_score'.format(dataset_name), dice_train, train_step)
                writer.add_scalar('Train/{}/IOU'.format(dataset_name), iou_train, train_step)

            # backward
            multi_loss = sum(datas_loss_list)
            optimizer.zero_grad()
            multi_loss.backward()
            optimizer.step()

            # logging average per batch
            writer.add_scalar('Train/Average/sum_loss',multi_loss.item(), train_step)
            writer.add_scalar('Train/Average/Di_score', sum(dice_train_list)/len(dice_train_list), train_step)
            writer.add_scalar('Train/Average/IOU', sum(iou_train_list)/len(iou_train_list), train_step)
            
            # end one training batch
            if config.debug: break

            # print
        print('Epoch {}, Total train step {} || sum_loss: {}, Avg Dice score: {}, Avg IOU: {}'.
        format(epoch, train_step, round(multi_loss.item(),5), round(sum(dice_train_list)/len(dice_train_list),4), 
        round(sum(iou_train_list)/len(iou_train_list),4)))
        print('Datasets: ', config.data.name, ' || loss: ', [round(x.item(), 4) for x in datas_loss_list], 
        ' || Dice score: ', [round(x, 4) for x in dice_train_list],
            ' || IOU: ', [round(x, 4) for x in iou_train_list])
            


        # -----------------------------------------------------------------
        # validate
        # ----------------------------------------------------------------
        model.eval()
        dice_val_list = []  # record results for each dataset
        iou_val_list = []
        loss_val_list = [] 
        # eval each dataset
        for dataset_name in config.data.name:
            dice_val_sum= 0
            iou_val_sum = 0
            loss_val_sum = 0
            num_val = 0
            for batch_id, batch in enumerate(val_loaders[dataset_name]):
                img = batch['image'].cuda().float()
                label = batch['label'].cuda().float()
                domain_label = batch['set_id']
                d = str(domain_label[0].item())
                domain_label = torch.nn.functional.one_hot(domain_label, 4).float().cuda()
                batch_len = img.shape[0]

                with torch.no_grad():
                    if config.model == 'BASE_DSN' and config.model_adapt.adapt_method == 'Sup':
                        output = model(img,domain_label=domain_label,d=d)
                    elif config.model_adapt.adapt_method and 'Sup' in config.model_adapt.adapt_method:
                        if config.model_adapt.Sup_label == 'Domain':
                            output = model(img, domain_label)
                    elif config.model=='BASE_DSN' or config.model_adapt.adapt_method in ['series_adapters', 'parallel_adapters']:
                        output = model(img,d=d)
                    else:
                        output = model(img)
                    output = torch.sigmoid(output)

                    # calculate loss
                    assert (output.shape == label.shape)
                    losses = []
                    for function in criterion:
                        losses.append(function(output, label))
                    loss_val_sum += sum(losses)*batch_len

                    # calculate metrics
                    output = output.cpu().numpy() > 0.5
                    label = label.cpu().numpy()
                    dice_val_sum += metrics.dc(output, label)*batch_len
                    iou_val_sum += metrics.jc(output, label)*batch_len

                    num_val += batch_len
                    # end one val batch
                    if config.debug: break

            # logging per epoch for one dataset
            loss_val_epoch, dice_val_epoch, iou_val_epoch = loss_val_sum/num_val, dice_val_sum/num_val, iou_val_sum/num_val
            dice_val_list.append(dice_val_epoch)
            loss_val_list.append(loss_val_epoch.item())
            iou_val_list.append(iou_val_epoch)
            writer.add_scalar('Val/{}/loss'.format(dataset_name), loss_val_epoch.item(), epoch)
            writer.add_scalar('Val/{}/Di_score'.format(dataset_name), dice_val_epoch, epoch)
            writer.add_scalar('Val/{}/IOU'.format(dataset_name), iou_val_epoch, epoch)

        # logging average per epoch
        writer.add_scalar('Val/Average/sum_loss', sum(loss_val_list), epoch)
        writer.add_scalar('Val/Average/Di_score', sum(dice_val_list)/len(dice_val_list), epoch)
        writer.add_scalar('Val/Average/IOU', sum(iou_val_list)/len(iou_val_list), epoch)
        # print
        print('Epoch {}, Validation || sum_loss: {}, Avg Dice score: {}, Avg IOU: {}'.
                format(epoch, round(sum(loss_val_list),5), 
                round(sum(dice_val_list)/len(dice_val_list),4), round(sum(iou_val_list)/len(iou_val_list),4)))
        print('Datasets: ', config.data.name, ' || loss: ', [round(x, 4) for x in loss_val_list], 
        ' || Dice score: ', [round(x, 4) for x in dice_val_list],
         ' || IOU: ', [round(x, 4) for x in iou_val_list])


        # scheduler step, record lr
        writer.add_scalar('Lr', scheduler.get_last_lr()[0], epoch)
        scheduler.step()

        # store model using the average iou
        avg_val_iou_epoch = sum(iou_val_list)/len(iou_val_list)
        if avg_val_iou_epoch > max_iou:
            torch.save(model.state_dict(), best_model_dir)
            max_iou = avg_val_iou_epoch
            best_epoch = epoch
            print('New best epoch {}!==============================='.format(epoch))
        
        end = time.time()
        time_elapsed = end-start
        print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))

        # end one epoch
        if config.debug: return
    
    print('Complete training ---------------------------------------------------- \n The best epoch is {}'.format(best_epoch))

    return 




# ========================================================================================================
def test(config, model, model_dir, test_loaders, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_list = []  # record results for each dataset
    iou_test_list = []
    loss_test_list = [] 
    # test each dataset
    for dataset_name in config.data.name:
        dice_test_sum= 0
        iou_test_sum = 0
        loss_test_sum = 0
        num_test = 0
        for batch_id, batch in enumerate(test_loaders[dataset_name]):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            domain_label = batch['set_id']
            d = str(domain_label[0].item())
            domain_label = torch.nn.functional.one_hot(domain_label, 4).float().cuda()
            batch_len = img.shape[0]
            with torch.no_grad():
                if config.model == 'BASE_DSN' and config.model_adapt.adapt_method == 'Sup':
                    output = model(img,domain_label=domain_label,d=d)
                elif config.model_adapt.adapt_method and 'Sup' in config.model_adapt.adapt_method:
                    if config.model_adapt.Sup_label == 'Domain':
                        output = model(img, domain_label)
                elif config.model=='BASE_DSN' or config.model_adapt.adapt_method in ['series_adapters', 'parallel_adapters']:
                    output = model(img,d=d)
                else:
                    output = model(img)  
                output = torch.sigmoid(output)

                # calculate loss
                assert (output.shape == label.shape)
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss_test_sum += sum(losses)*batch_len

                # calculate metrics
                output = output.cpu().numpy() > 0.5
                label = label.cpu().numpy()
                dice_test_sum += metrics.dc(output, label)*batch_len
                iou_test_sum += metrics.jc(output, label)*batch_len

                num_test += batch_len
                # end one test batch
                if config.debug: break

        # logging results for one dataset
        loss_test_epoch, dice_test_epoch, iou_test_epoch = loss_test_sum/num_test, dice_test_sum/num_test, iou_test_sum/num_test
        dice_test_list.append(dice_test_epoch)
        loss_test_list.append(loss_test_epoch.item())
        iou_test_list.append(iou_test_epoch)


    # logging average and store results
    dataset_name_list = config.data.name+['Total']
    loss_test_list.append(sum(loss_test_list))
    dice_test_list.append(sum(dice_test_list)/len(dice_test_list))
    iou_test_list.append(sum(iou_test_list)/len(iou_test_list))
    df = pd.DataFrame({
        'Name': dataset_name_list,
        'loss': loss_test_list,
        'Di_score': dice_test_list,
        'IOU': iou_test_list
    })
    df.to_csv(test_results_dir, index=False)

    # print
    print('========================================================================================')
    print('Test || Average loss: {}, Dice score: {}, IOU: {}'.
            format(round(sum(loss_test_list),5), 
            round(sum(dice_test_list)/len(dice_test_list),4), round(sum(iou_test_list)/len(iou_test_list),4)))
    print('Datasets: ', config.data.name, ' || loss: ', [round(x, 4) for x in loss_test_list], 
    ' || Dice score: ', [round(x, 4) for x in dice_test_list], ' || IOU: ', [round(x, 4) for x in iou_test_list])

    return




if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp_name', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str,default='Configs/multi_train_local.yml')
    parser.add_argument('--model', type=str,default='DeepResUnet')
    parser.add_argument('--select_patch', action='store_true') # for DeepRUST model. True means using selective patch. If use this means ture
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='isic2018')
    parser.add_argument('--k_fold', type=str, default='No')
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['model'] = args.model
    config['train']['batch_size']=args.batch_size
    config['select_patch'] = args.select_patch
    config['data']['name'] = args.dataset
    config['model_adapt']['adapt_method']=args.adapt_method
    config['data']['k_fold'] = args.k_fold

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))
    
    store_config = config
    config = DotDict(config)
    
    # logging tensorbord, config, best model
    exp_dir = '{}/results/{}_{}_{}'.format(config.root_dir,args.exp_name,config.model,now.strftime("%Y%m%d_%H%M"))
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(exp_dir)
    best_model_dir = '{}/best.pth'.format(exp_dir)
    test_results_dir = '{}/test_results.csv'.format(exp_dir)

    # store yml file
    if config.debug == False:
        yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
    
    # torch.set_num_threads(8)
    if config.model_adapt.Sup_label == 'Domain':
        num_list = [2594, 200, 1212, 206]
    elif config.model_adapt.Sup_label == 'DC':
        num_list = [519, 1867, 208, 80, 80, 40, 123, 239, 257, 65, 331, 76, 121, 87, 119]
    K = len(num_list)  # num of domains

    main(config)