import torch
import torch.nn as nn
import numpy as np
from .losses import Custom_MLSM_Loss

def test(model,testloader,device):
#       cnvt=config.cnvt
      model.eval()
      num_steps = 0
      val_loss_total = 0.0
      final_loss=[]
      tool_wts=torch.from_numpy(np.array([0.08417016, 0.86882581, 0.11139783, 2.62039429, 1.65233037, 1.17782675])).to(device)
      verb_wts=torch.from_numpy(np.array([ 0.32652701,  0.07326159,  0.08546381,  0.80298039,  1.32513591,  2.30630946,  1.36341827,  7.38269082, 16.8797437,   0.4518677 ])).to(device)
      target_wts=torch.from_numpy(np.array([ 0.07638518,  1.,         0.48500517,  0.93240071, 17.62857, 11.33425635, 1.58219869,  5.49887714 , 0.33932505, 23.18846306,  0.72411939,  4.26167666,  7.48678624,  0.91095904,  0.52437649])).to(device)
  
      triplet_accuracy=0.0
      i_ac=0.0
      v_ac=0.0
      t_ac=0.0
      mat_ac=0.0  
      triplet_total=0.0
      i_total=0.0
      v_total=0.0
      t_total=0.0
      mat_total=0.0

      sig=nn.Sigmoid()


      eps=1e-12  


      criterion=Custom_MLSM_Loss()

      for batch_idx, (data,tars,tars_i,tars_v,tars_t) in enumerate(testloader):


          data, tars,tars_i,tars_v,tars_t = data.to(device), tars.to(device), tars_i.to(device), tars_v.to(device), tars_t.to(device)

          bs=data.shape[0]
          with torch.cuda.amp.autocast(enabled=True):
                output,output_i,output_v,output_t,_,__,___,_ = model(data)

                loss_instrument=criterion(target=tars_i,input=output_i,wts=tool_wts)
                loss_target=criterion(target=tars_t,input=output_t,wts=target_wts)
                loss_verb=criterion(target=tars_v,input=output_v,wts=verb_wts) 
                loss_triplet=criterion(target=tars,input=output)




                loss=loss_instrument+loss_triplet+loss_verb+loss_target

          val_loss_total += loss.item()
          final_loss.append(loss.item())


          triplet_accuracy=(torch.sum(torch.round(sig(output.type(torch.float32).detach().cpu()))*tars.detach().cpu())/(torch.sum(tars.detach().cpu())+eps))/bs
          i_ac=(torch.sum(torch.round(sig(output_i.type(torch.float32).detach().cpu()))*tars_i.detach().cpu())/(torch.sum(tars_i.detach().cpu())+eps))/bs
          v_ac=(torch.sum(torch.round(sig(output_v.type(torch.float32).detach().cpu()))*tars_v.detach().cpu())/(torch.sum(tars_v.detach().cpu())+eps))/bs
          t_ac=(torch.sum(torch.round(sig(output_t.type(torch.float32).detach().cpu()))*tars_t.detach().cpu())/(torch.sum(tars_t.detach().cpu())+eps))/bs

          triplet_total+=triplet_accuracy
          i_total+=i_ac
          v_total+=v_ac
          t_total+=t_ac 




          num_steps += 1


      val_loss_total_avg = val_loss_total/num_steps  
      triplet_total=triplet_total/num_steps
      t_total=t_total/num_steps  
      i_total=i_total/num_steps
      v_total=v_total/num_steps

      print(' Test Val Loss: {:.4f}'.format(val_loss_total_avg))        
      return final_loss