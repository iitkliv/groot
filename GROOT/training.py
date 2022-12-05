import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import datetime
import warnings

from train_utils.get_config import get_config
from train_utils.model import get_model
from train_utils.transforms import transforms
from train_utils.dataloader import custom_data_loader
from train_utils.train_function import train
from train_utils.test_function import test


def setup_parameters(config):

    use_cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA', use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")    
    
    
    dataset_path = config["data"]["dataset_path"]

    data_path = os.path.join(dataset_path, 'data')
    triplet_path = os.path.join(dataset_path, 'triplet')
    instrument_path = os.path.join(dataset_path, 'instrument')
    verb_path = os.path.join(dataset_path, 'verb')
    target_path = os.path.join(dataset_path, 'target')
    dict_path = os.path.join(dataset_path, 'dict')
    video_names = os.listdir(data_path)                                   

    print("Dataset paths successfully defined!")

    with open(os.path.join(dict_path, 'instrument.txt'), 'r') as f:
      instrument_info = f.readlines()
      instrument_dict = {}
      for l in instrument_info:
        instrument_id, instrument_label = l.split(':')
        instrument_dict[instrument_label.rstrip()] = int(instrument_id)

    with open(os.path.join(dict_path, 'verb.txt'), 'r') as f:
      verb_info = f.readlines()
      verb_dict = {}
      for l in verb_info:
        verb_id, verb_label = l.split(':')
        verb_dict[verb_label.rstrip()] =int(verb_id) 

    with open(os.path.join(dict_path, 'target.txt'), 'r') as f:
      target_info = f.readlines()
      target_dict = {}
      for l in target_info:
        target_id, target_label = l.split(':')
        target_dict[target_label.rstrip()] = int(target_id)

    def get_instrument_triplet_mapping(dict_path):
#           dict_path='/home/raviteja/Cholec_Triplet/CholecT50-challenge-train/dict/'#'/home/raviteja/Cholec_Triplet/docker_22/'+
          with open(os.path.join(dict_path, 'maps.txt'), 'r') as f:
              target_info = f.readlines()
              target_dict = {}
              for l in target_info[1:]:
                ids=l.split(',')
                target_dict[int(ids[0])] = [int(i) for i in ids[1:]]

          return target_dict

    maps_dict=get_instrument_triplet_mapping(dict_path)

    if (config["config"]["init"]=="KHe"):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight)
            if type(m) == nn.Conv2d:
                nn.init.kaiming_uniform_(m.weight)
            if type(m) == torch.nn.parameter.Parameter:
                nn.init.kaiming_uniform_(m)


    if (config["config"]["init"]=="Xa"):
        def init_weights(m):    
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == torch.nn.parameter.Parameter:
                nn.init.xavier_uniform_(m) 


    # Create dictionary mapping triplet ids to readable label

    with open(os.path.join(dict_path, 'triplet.txt'), 'r') as f:
      triplet_info = f.readlines()
      triplet_dict = {}
      for l in triplet_info:
        triplet_id, triplet_label = l.split(':')
        triplet_dict[int(triplet_id)] = triplet_label.rstrip()

    print('Random triplet id and its human readable label\n')
    random_triplet_id = np.random.choice(list(triplet_dict.keys()))
    print('Triplet id: ', random_triplet_id, '\nReadable label: ', triplet_dict[random_triplet_id])



    train_videos=np.load(dataset_path+'train_videos.npy')
    test_videos=np.load(dataset_path+'test_videos.npy')
    train_files=np.load(dataset_path+'train_files.npy')
    test_files=np.load(dataset_path+'test_files.npy')

    train_triplet=np.load(dataset_path+'train_triplet.npy',allow_pickle=True)
    test_triplet=np.load(dataset_path+'test_triplet.npy',allow_pickle=True)


    train_instrument=np.load(dataset_path+'train_instrument.npy',allow_pickle=True)
    test_instrument=np.load(dataset_path+'test_instrument.npy',allow_pickle=True)


    train_verb=np.load(dataset_path+'train_verb.npy',allow_pickle=True)
    test_verb=np.load(dataset_path+'test_verb.npy',allow_pickle=True)


    train_target=np.load(dataset_path+'train_target.npy',allow_pickle=True)
    test_target=np.load(dataset_path+'test_target.npy',allow_pickle=True)

    transforms_obj=transforms()
    
    trainloader = torch.utils.data.DataLoader(custom_data_loader(maps_dict=maps_dict,x_list = train_files, y_list = train_triplet.tolist(),y_instrument=train_instrument.tolist(),y_verb=train_verb.tolist(),y_target=train_target.tolist(),norm=eval(config["data"]["norm"]),transforms=transforms_obj,aug=eval(config["config"]["aug"]),crop=eval(config["config"]["crop"]), crop_shape=(256,256),resize = eval(config["config"]["resize"]), resize_shape=eval(config["config"]["resize_shape"])), batch_size=config["config"]["tr_batch_size"], num_workers=config["config"]["nworkers"], shuffle = eval(config["config"]["shuffle"]), pin_memory=True)

    testloader = torch.utils.data.DataLoader(custom_data_loader(maps_dict=maps_dict,x_list = test_files, y_list = test_triplet.tolist(),y_instrument=test_instrument.tolist(),y_verb=test_verb.tolist(),y_target=test_target.tolist(),norm=eval(config["data"]["norm"]),transforms=transforms_obj , aug=False,crop=eval(config["config"]["crop"]), crop_shape=(256,256),resize = eval(config["config"]["resize"]), resize_shape=eval(config["config"]["resize_shape"])), batch_size=config["config"]["val_batch_size"], num_workers=config["config"]["nworkers"], shuffle = eval(config["config"]["shuffle"]), pin_memory=True)
    
    
    
    
    adj_mat_ins=torch.from_numpy(np.load(dataset_path+'adj_mat_ins_intra.npy'))
    adj_mat_ins=adj_mat_ins.to(torch.float32)#.to(device)

    adj_mat_verb=torch.from_numpy(np.load(dataset_path+'adj_mat_verb_intra.npy'))
    adj_mat_verb=adj_mat_verb.to(torch.float32)#.to(device)

    adj_mat_tar=torch.from_numpy(np.load(dataset_path+'adj_mat_tar_intra.npy'))
    adj_mat_tar=adj_mat_tar.to(torch.float32)#.to(device)

    adj_mat_trip=torch.from_numpy(np.load(dataset_path+'adj_mat_trip_intra.npy'))
    adj_mat_trip=adj_mat_trip.to(torch.float32)#.to(device)

    adj_mat_inter=torch.from_numpy(np.load(dataset_path+'adj_mat_inter.npy'))
    adj_mat_inter=adj_mat_inter.to(torch.float32)#.to(device)

    adj_mat=[adj_mat_ins,adj_mat_verb,adj_mat_tar,adj_mat_inter[:31,31:],adj_mat_trip]

    
    model=get_model(config["classes"]["c_i"],config["classes"]["c_v"],config["classes"]["c_t"],config["classes"]["c_ivt"],adj_mat,config["ssl"],device)
    
    
    model.sam_i=model.sam_i.apply(init_weights)
    model.sam_v=model.sam_v.apply(init_weights)
    model.sam_t=model.sam_t.apply(init_weights)
    model.gcn_i=model.gcn_i.apply(init_weights) 
    model.gcn_v=model.gcn_v.apply(init_weights) 
    model.gcn_t=model.gcn_t.apply(init_weights) 
    model.gcn_triplet_inter=model.gcn_triplet_inter.apply(init_weights)
    model.head_i=model.head_i.apply(init_weights)
    model.head_v=model.head_v.apply(init_weights)
    model.head_t=model.head_t.apply(init_weights)
    model.head_triplet=model.head_triplet.apply(init_weights)
    model.cam_input_mixer=model.cam_input_mixer.apply(init_weights)
    model.backbone_replica=model.backbone_replica.apply(init_weights)


    scaler = torch.cuda.amp.GradScaler()




    
    decay_rate=config["config"]["decay_rate"]
    initial_lr_bb = config["config"]['learning_rate_bb']
    initial_lr_ins = config["config"]["learning_rate_ins"]
    
    optimizer = optim.Adam([{'params':model.backbone.parameters(),'lr':initial_lr_bb},
                              {'params':model.sam_i.parameters(),'lr':initial_lr_ins},
                              {'params':model.gcn_i.parameters(),'lr':initial_lr_ins},
    
                              {'params':model.head_i.parameters(),'lr':initial_lr_ins},
                              {'params':model.sam_v.parameters(),'lr':initial_lr_ins},
                              {'params':model.gcn_v.parameters(),'lr':initial_lr_ins},
    
                              {'params':model.head_v.parameters(),'lr':initial_lr_ins},
                              {'params':model.sam_t.parameters(),'lr':initial_lr_ins},
                              {'params':model.gcn_t.parameters(),'lr':initial_lr_ins},
    
                              {'params':model.head_t.parameters(),'lr':initial_lr_ins},                
                              {'params':model.gcn_triplet_inter.parameters(),'lr':initial_lr_ins},
    
                              {'params':model.head_triplet.parameters(),'lr':initial_lr_ins},
                              {'params':model.cam_input_mixer.parameters(),'lr':initial_lr_ins},
                              {'params':model.backbone_replica.parameters(),'lr':initial_lr_ins}],lr=initial_lr_bb)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer,decay_rate)#
    
    parameter_dict={
    "trainloader":trainloader,
    "testloader":testloader,
    "optimizer":optimizer,
    "scaler":scaler,
    "model" :model,
    "device":device,
    "resize_shape":eval(config["config"]["resize_shape"]),  
    "checkpoint":config["config"]["checkpoint"],
    "epochs" : config["config"]["epochs"],
    "basepath": config["output"]["base_path"],
    "exp_id":config["config"]["exp_id"],
    "exp_no":config["config"]["exp_no"],    
    }
    
    return parameter_dict







def train_model(config):
    

    exp_path=parameter_dict["basepath"]+parameter_dict["exp_id"]+'/'
    model_path=exp_path+'models/'
    stats_path=exp_path+'stats/'
    method=parameter_dict["exp_id"]+'_'+parameter_dict["exp_no"]
    
    
    
    model=parameter_dict["model"]
    optimizer=parameter_dict["optimizer"]
    scaler=parameter_dict["scaler"]
    trainloader=parameter_dict["trainloader"]
    testloader=parameter_dict["testloader"]
    device=parameter_dict["device"]
    resize_shape=parameter_dict["resize_shape"]
    checkpoint=parameter_dict["checkpoint"]
    epochs = parameter_dict["epochs"]
    
    
    model=nn.DataParallel(model)
    model=model.to(device)
    start=datetime.datetime.now()
    total_loss=[]
    avg_loss=[]
    test_loss=[]
    b_loss=[]
    batchloss=[]

    for epoch in range(1, epochs +1):

      avg_loss,b_loss=train(epoch,model,trainloader,scaler,optimizer,resize_shape,device)

      total_loss.extend(avg_loss)
      batchloss.extend(b_loss)
      with torch.no_grad(): test_loss.extend(test(model,testloader,device))


      if(epoch % checkpoint ==0):
        torch.save(model.state_dict(),model_path+method+"_ep_"+str(epoch)+".pth")
        np.save(stats_path+method+"_ep_"+str(epoch)+'_train_loss',total_loss)
        np.save(stats_path+method+"_ep_"+str(epoch)+'_val_loss',test_loss)
        torch.save(optimizer.state_dict(),model_path+method+"_optimizer_ep_"+str(epoch)+".pth")


    end=datetime.datetime.now()

    print('run time=',end-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='absolute path to config files', type=str)
    args = parser.parse_args()
    
    config_dict = get_config(config_path=args.path)
    parameter_dict=setup_parameters(config_dict)
    train_model(parameter_dict)  