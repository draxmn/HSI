from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import  os
import time
import scipy.io as sio
import torch.utils.data as data
from dataset import DatasetFromHdf5
from resblock import resblock,conv_relu_res_relu_block
from utils import AverageMeter,initialize_logger,save_checkpoint,record_loss
from loss import rrmse_loss
import gc
device = torch.device('cuda')

#import GPUtil
#GPUtil.showUtilization()


def main():
    #https://drive.google.com/file/d/1QxQxf2dzfSbvCgWlI9VuxyBgfmQyCmfE/view?usp=sharing - train data
    #https://drive.google.com/file/d/11INkjd_ajT-RSCSFqfB7reLI6_m1jCAC/view?usp=sharing - val data
    #https://drive.google.com/file/d/1m0EZaRjla2o_eL3hOd7UMkSwoME5mF4A/view?usp=sharing - extra val data
    cudnn.benchmark = True
  #  train_data = DatasetFromHdf5('C:/Users/alawy/Desktop/Training/Training-shadesofgrey/train_tbands.h5')
    train_data = DatasetFromHdf5('/storage/train_cropped14.h5')

    print(len(train_data))
    val_data_extra = DatasetFromHdf5('/storage/valid_extra99.h5')
    val_data = DatasetFromHdf5('/storage/valid_cropped89.h5')
    new_val=[]
    new_val.append(val_data)
    new_val.append(val_data_extra)
    print(len(new_val))
    print('con')
    val_new = data.ConcatDataset(new_val)
    print(len(val_new))

    # Data Loader (Input Pipeline)
    train_data_loader = DataLoader(dataset=train_data, 
                                   num_workers=4,  
                                   batch_size=512,
                                   shuffle=True,
                                   pin_memory=True)
    val_loader = DataLoader(dataset=val_new,
                            num_workers=1, 
                            batch_size=1,
                            shuffle=False,
                           pin_memory=True)
    # Dataset
   # torch.set_num_threads(12)
    # Model               
    model = resblock(conv_relu_res_relu_block, 16, 3, 25)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.to('cuda')
    # Parameters, Loss and Optimizer
    start_epoch = 0
    end_epoch = 1000
    init_lr = 0.0002
    iteration = 0
    record_test_loss = 1000
    criterion = rrmse_loss
    #optimizer=torch.optim.AdamW(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer=torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
   # model_path = '/storage/models-crop/'
    model_path = './models-crop/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path,'loss.csv'), 'w+')
    
    log_dir = os.path.join(model_path,'train.log')
    logger = initialize_logger(log_dir)
    
    # Resume
    resume_file = ''
    #resume_file = '/storage/notebooks/r9h1kyhq8oth90j/models/hscnn_5layer_dim10_69.pkl' 
    #resume_file = '/storage/notebooks/r9h1kyhq8oth90j/models-crop/hscnn_5layer_dim10_95.pkl'
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
       
    for epoch in range(start_epoch+1, end_epoch):
        
        start_time = time.time()         
        train_loss, iteration, lr = train(train_data_loader, model, criterion, optimizer, iteration, init_lr, end_epoch)
        test_loss = validate(val_loader, model, criterion)
        
 
        
        # Save model
        if test_loss < record_test_loss:
            record_test_loss = test_loss
            save_checkpoint(model_path, epoch, iteration, model, optimizer)
        else:
            save_checkpoint(model_path, epoch, iteration, model, optimizer)
        # print loss 
        end_time = time.time()
        epoch_time = end_time - start_time
        print ("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, test_loss))
        # save loss
        record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, test_loss)     
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, test_loss))
        gc.collect()
# Training 
def train(train_data_loader, model, criterion, optimizer, iteration, init_lr ,end_epoch):
    losses = AverageMeter()
    for i, (images, labels) in enumerate(train_data_loader):
        labels = labels.to('cuda')
        images = images.to('cuda')
        images = Variable(images)
        labels = Variable(labels) 
        # Decaying Learning Rate
        
        lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=1.5) 
        iteration = iteration + 1
        # Forward + Backward + Optimize       
        output = model(images)
        
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        
        #  record loss
        losses.update(loss.data)
        #losses.update(loss.item())
    return losses.avg, iteration, lr

# Validate
def validate(val_loader, model, criterion):
    
    
    model.eval()
    losses = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.to('cuda')
        target = target.to('cuda')
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)      
        loss = criterion(output, target_var)

        #  record loss
        losses.update(loss.item())

    return losses.avg

# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
