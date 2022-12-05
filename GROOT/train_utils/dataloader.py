import os
import cv2
import torch
import torchvision
from torch.utils import data
import numpy as np
import torchvision.transforms as tt


class custom_data_loader(data.Dataset):

      def __init__(self,maps_dict, x_list, y_list,y_instrument,y_verb,y_target,norm,transforms,aug = True, crop=False,crop_shape = (256,256), resize=True, resize_shape=[256,256]):

            self.x_list =x_list
            self.y_list=y_list
            self.y_instrument=y_instrument
            self.y_verb=y_verb
            self.y_target=y_target        
            self.transforms=transforms
            self.aug = aug
            self.crop=crop
            self.crop_shape = crop_shape
            self.resize = resize
            self.resize_shape = resize_shape
            self.maps_dict=maps_dict
            self.norm=norm


      def __len__(self):
            return len(self.x_list)

      def __getitem__(self, index):

          image_file_name=self.x_list[index]      
          image = None
          
          if os.path.isfile(image_file_name):
              image=cv2.imread(image_file_name)
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              img_shape=image.shape

          else:
              print('ERROR: couldn\'t find image -> ', image_file_name)
          if image is None:
              print('ERROR: couldn\'t find image -> ', image_file_name)






          label_ref=image_file_name.split('/')
          label_id=label_ref[-2]+label_ref[-1].split('.')[0]

          label=self.y_list[label_id]    
          label_i=self.y_instrument[label_id]     
          label_v=self.y_verb[label_id]     
          label_t=self.y_target[label_id]     

    
          if label is None:
              print('ERROR: couldn\'t find label -> ', label_id)
    #----------------------------------------------------------------------------------------------------------------------------------------
    # resize to 240x427 and convert back to np 
          if self.resize:
              rs=(self.resize_shape[0],self.resize_shape[1])
              im=torch.from_numpy(image)
              im=im.permute(2,0,1)
              rsize=tt.Resize(rs,interpolation=tt.InterpolationMode.BILINEAR)
              img2=rsize(im)
              image=img2.permute(1,2,0).numpy()
    #----------------------------------------------------------------------------------------------------------------------------------------     
    
          if self.aug:

              transformed=self.transforms(image=image)
              image = transformed["image"]
    #----------------------------------------------------------------------------------------------------------------------------------------



          return self.transform_image(image), torch.from_numpy(label),torch.from_numpy(label_i),torch.from_numpy(label_v),torch.from_numpy(label_t)#,label_tr_id,label_tr_i,label_tr_v,label_tr_t
    

      def transform_image(self, image):
          image = image.astype(np.uint8)


          normalize = tt.Compose([
                tt.ToTensor(), 
                tt.Normalize(self.norm[0], self.norm[1]),
            ])
          image=normalize(image) 

          return image

      def one_hot_encoder(self,target):
        num_class=2
        t_shape=target.shape
        target=torch.floor_(target[0,:,:])
        temp_target=torch.tensor(target,dtype=int)
        
        encoded_target=torch.nn.functional.one_hot(temp_target,num_class)
        
        encoded_target=encoded_target.permute(2,0,1)
        encoded_target=encoded_target.reshape(num_class,t_shape[1],t_shape[2])
        return encoded_target





    