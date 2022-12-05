
import torch
import torchvision.transforms as tt
import numpy as np
from .losses import Custom_MLSM_Loss


def train(epoch,model,trainloader,scaler,optimizer,resize_shape,device, log_interval=500):
    
        tool_wts=torch.from_numpy(np.array([0.08417016, 0.86882581, 0.11139783, 2.62039429, 1.65233037, 1.17782675])).to(device)
        verb_wts=torch.from_numpy(np.array([ 0.32652701,  0.07326159,  0.08546381,  0.80298039,  1.32513591,  2.30630946,  1.36341827,  7.38269082, 16.8797437,   0.4518677 ])).to(device)
        target_wts=torch.from_numpy(np.array([ 0.07638518,  1.,         0.48500517,  0.93240071, 17.62857, 11.33425635, 1.58219869,  5.49887714 , 0.33932505, 23.18846306,  0.72411939,  4.26167666,  7.48678624,  0.91095904,  0.52437649])).to(device)


        rs=(resize_shape[0],resize_shape[1])
        rsize=tt.Resize(rs,interpolation=tt.InterpolationMode.BICUBIC)

#         cnvt=config["config"]["cnvt"]
        model.train()

        avg_loss=0.0
        total_loss=[]
        b_loss=[]
    
        criterion=Custom_MLSM_Loss()
    
        for batch_idx, (data,tars,tars_i,tars_v,tars_t) in enumerate(trainloader):

            data, tars,tars_i,tars_v,tars_t = data.to(device), tars.to(device), tars_i.to(device), tars_v.to(device), tars_t.to(device)
            optimizer.zero_grad()
    
            with torch.cuda.amp.autocast(enabled=True):

                #pass1
                output,output_i,output_v,output_t,cam_i,cam_v,cam_t,cam_trip = model(data)

                loss_instrument=criterion(target=tars_i,input=output_i,wts=tool_wts)
                loss_target=criterion(target=tars_t,input=output_t,wts=target_wts)
                loss_verb=criterion(target=tars_v,input=output_v,wts=verb_wts) 
                loss_triplet=criterion(target=tars,input=output)
                MLSM_losses=loss_instrument+loss_triplet+loss_verb+loss_target



                cam_i=torch.einsum('bc, bcij->bcij', tars_i, rsize(cam_i))
                cam_v=torch.einsum('bc, bcij->bcij', tars_v, rsize(cam_v))
                cam_t=torch.einsum('bc, bcij->bcij', tars_t, rsize(cam_t))
                cam_trip=torch.einsum('bc, bcij->bcij', tars, rsize(cam_trip))

                output2,output_i2,output_v2,output_t2 = model(data,cam_i.detach(),cam_v.detach(),cam_t.detach(),cam_trip.detach(),second_stage_flag=True)

                loss_instrument2=criterion(target=tars_i,input=output_i2,wts=tool_wts)
                loss_target2=criterion(target=tars_t,input=output_t2,wts=target_wts)
                loss_verb2=criterion(target=tars_v,input=output_v2,wts=verb_wts) 
                loss_triplet2=criterion(target=tars,input=output2)
                MLSM_losses2=loss_instrument2+loss_triplet2+loss_verb2+loss_target2            




                #rmse custom loss
                rmse_loss=torch.sqrt(torch.mean((output-output2)**2))+torch.sqrt(torch.mean((output_i-output_i2)**2))+torch.sqrt(torch.mean((output_v-output_v2)**2))+torch.sqrt(torch.mean((output_t-output_t2)**2))



                loss = MLSM_losses+MLSM_losses2+rmse_loss
    ##############################################completed stage 2 till here code below for loss=--> 

            avg_loss+=loss.item()
            total_loss.append(loss.item())         
            scaler.scale(loss).backward(inputs=list(filter(lambda p: p.requires_grad, model.parameters())))

            scaler.step(optimizer)

            scaler.update() 


            if batch_idx % log_interval == 0 and batch_idx > 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader),avg_loss/(batch_idx)))


        b_loss.append(avg_loss/(batch_idx+1))

        return total_loss,b_loss


