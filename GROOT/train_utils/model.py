import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import copy
from .get_config import get_config



class Instrument_SAM(nn.Module):
    def __init__(self,num_instruments,in_features,out_features):
        super(Instrument_SAM, self).__init__()
        self.fc_3i= nn.Sequential(nn.Conv2d(in_features, in_features, (3,3), bias=True),
                                   nn.BatchNorm2d(in_features),
                                   nn.ReLU())
        self.fc_i = nn.Conv2d(in_features, num_instruments, (1,1), bias=False)
        self.conv_transform = nn.Conv2d(in_features, out_features, (1,1))#nn.Sequential(,
#                               nn.BatchNorm2d(out_features),
#                                    nn.ReLU())
         #infeatures=512,outfeatures=256, for resnet18 
    def forward(self,x):
      with torch.cuda.amp.autocast(enabled=True):
        x =    self.fc_3i(x)
        mask = self.fc_i(x)
        mask_size=mask.shape
        mask = mask.view(mask.size(0), mask.size(1), -1) 

        out1=mask.topk(1, dim=-1)[0].mean(dim=-1) #top k returns b*ci*1 and then mean makes out1 => b*ci

        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask) #x => b*out_features*c_i 
        mask=mask.transpose(1, 2)
        mask=mask.view(mask_size[0],mask_size[1],mask_size[2],mask_size[3])
        return x,out1,mask

class Verb_SAM(nn.Module):
    def __init__(self,num_verbs,in_features,out_features):
        super(Verb_SAM, self).__init__()
        self.fc_3v= nn.Sequential(nn.Conv2d(in_features, in_features, (3,3), bias=True),
                                   nn.BatchNorm2d(in_features),
                                   nn.ReLU())
        self.fc_v = nn.Conv2d(in_features+21, num_verbs, (1,1), bias=False)
        self.conv_transform = nn.Conv2d(in_features, out_features, (1,1))
#                               nn.Sequential(nn.Conv2d(in_features, out_features, (1,1)),
#                               nn.BatchNorm2d(out_features),
#                                    nn.ReLU())
         #infeatures=512,outfeatures=256, for resnet18 
    def forward(self,x,instrument_features,target_features):
      with torch.cuda.amp.autocast(enabled=True):

        x=self.fc_3v(x)
        fusued_features=torch.cat((x,instrument_features,target_features),dim=1)

        mask = self.fc_v(fusued_features) 

        mask_size=mask.shape
        mask = mask.view(mask.size(0), mask.size(1), -1) 

        out1=mask.topk(1, dim=-1)[0].mean(dim=-1) #top k returns b*ci*1 and then mean makes out1 => b*ci

        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask) #x => b*out_features*c_i 
        mask=mask.transpose(1, 2)
        mask=mask.view(mask_size[0],mask_size[1],mask_size[2],mask_size[3])

        return x,out1,mask        

class Target_SAM(nn.Module):
    def __init__(self,num_targets,in_features,out_features):
        super(Target_SAM, self).__init__()
        self.fc_3t = nn.Sequential(nn.Conv2d(in_features, in_features, (3,3), bias=True),
                                   nn.BatchNorm2d(in_features),
                                   nn.ReLU())
        self.fc_t = nn.Conv2d(in_features+6, num_targets, (1,1), bias=False)
        self.conv_transform = nn.Conv2d(in_features, out_features, (1,1))
#                               nn.Sequential(nn.Conv2d(in_features, out_features, (1,1)),
#                               nn.BatchNorm2d(out_features),
#                                    nn.ReLU())
         #infeatures=512,outfeatures=256, for resnet18 
    def forward(self,x,instrument_features):
      with torch.cuda.amp.autocast(enabled=True):
        x=self.fc_3t(x)
        fusued_features=torch.cat((x,instrument_features),dim=1)
        mask = self.fc_t(fusued_features) 
        mask_size=mask.shape
        mask = mask.view(mask.size(0), mask.size(1), -1) 

        out1=mask.topk(1, dim=-1)[0].mean(dim=-1) #top k returns b*ci*1 and then mean makes out1 => b*ci

        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask) #x => b*out_features*c_i 
        mask=mask.transpose(1, 2)
        mask=mask.view(mask_size[0],mask_size[1],mask_size[2],mask_size[3])

        return x,out1,mask                  

class DynamicGraphConvolution(nn.Module):    
  def __init__(self, in_features, out_features, num_nodes,device,stat_adj_mat=None,component=False):
        super(DynamicGraphConvolution, self).__init__()
        if stat_adj_mat is None: 
            self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.ReLU())    #numnodes =v1 =131
#             nn.LeakyReLU(0.2))
            self.fixed_adjmat_provided=False
        else: 
          self.static_adj=nn.Parameter(stat_adj_mat)
          self.fixed_adjmat_provided=True

        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, in_features, 1), #infeatures=outfeatures=256 #for ResNet18
            nn.ReLU())
#             nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)  #infeatures=outfeatures=256 #for ResNet18
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()#nn.LeakyReLU(0.2)
        self.component=component
        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)  #infeatures=outfeatures=256, nodes=131 #for ResNet18
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)   #infeatures=outfeatures=256 #for ResNet18        
        self.device=device    
  def forward_static_gcn(self, x):
    if self.fixed_adjmat_provided:
        adj_mat=self.static_adj.to(self.device).detach()   
        x = torch.matmul(x,adj_mat)
        x = x.transpose(1, 2)
        x = self.static_weight(x.transpose(1, 2))
        return x
    else:  
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

  def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
#         if self.component: dynamic_adj = torch.sigmoid(dynamic_adj)
#         else: 
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

  def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

  def forward(self, x,residue=True):
    with torch.cuda.amp.autocast(enabled=True):
        """ D-GCN module
        Shape: 
        - Input: (B, C_in, N) # C_in: 256, N: num_classes (131)
        - Output: (B, C_out, N) # C_out: 256, N: num_classes (131)
        """

        out_static = self.forward_static_gcn(x)
        if residue:  x = x + out_static # residual
        else: x=out_static
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x


class Instrument_Head(nn.Module):    
  def __init__(self, in_features, num_instruments):
        super(Instrument_Head, self).__init__()
        self.mask_mat = nn.Parameter(torch.eye(num_instruments).float())
        self.last_linear = nn.Conv1d(in_features, num_instruments, 1)

  def forward(self,z,out1):
    with torch.cuda.amp.autocast(enabled=True):
        out2 = self.last_linear(z) # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)

        return (out2+out1)/2

class Verb_Head(nn.Module):    
  def __init__(self, in_features, num_verbs):
        super(Verb_Head, self).__init__()
        self.mask_mat = nn.Parameter(torch.eye(num_verbs).float())
        self.last_linear = nn.Conv1d(in_features, num_verbs, 1)

  def forward(self,z,out1):
    with torch.cuda.amp.autocast(enabled=True):
        out2 = self.last_linear(z) # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)

        return (out2+out1)/2        

class Target_Head(nn.Module):    
  def __init__(self, in_features, num_targets):
        super(Target_Head, self).__init__()
        self.mask_mat = nn.Parameter(torch.eye(num_targets).float())
        self.last_linear = nn.Conv1d(in_features, num_targets, 1)

  def forward(self,z,out1):
    with torch.cuda.amp.autocast(enabled=True):
        out2 = self.last_linear(z) # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)

        return (out2+out1)/2 

class Triplet_Head(nn.Module):    
  def __init__(self, in_features, num_triplets):
        super(Triplet_Head, self).__init__()
        self.mask_mat = nn.Parameter(torch.eye(num_triplets).float())
        self.last_linear = nn.Conv1d(in_features, num_triplets, 1)

  def forward(self,z,out3):
    with torch.cuda.amp.autocast(enabled=True):
        out2 = self.last_linear(z) # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)

        return (out2+out3)/2 

class Triplet_ADD_GCN(nn.Module):
    def __init__(self,backbone,num_instruments,num_verbs,num_targets,num_triplets,device,stat_adj_mat=None):
        super(Triplet_ADD_GCN, self).__init__()
        self.backbone = backbone

#         self.cam_input_mixer=nn.Conv2d((3+num_instruments+num_verbs+num_targets), 3, (1,1))
        self.cam_input_mixer=nn.Conv2d((134), 3, (1,1))
        self.backbone_replica=copy.deepcopy(backbone)

        self.num_instruments=num_instruments
        self.num_verbs=num_verbs
        self.num_targets=num_targets
        self.num_triplets=num_triplets
        self.stat_adj_mat=stat_adj_mat

        self.sam_i=Instrument_SAM(num_instruments,2048,256)
        self.sam_v=Verb_SAM(num_verbs,2048,256)
        self.sam_t=Target_SAM(num_targets,2048,256)


        self.gcn_i = DynamicGraphConvolution(256, 256, 6,device,stat_adj_mat[0],component=True)
        self.gcn_v = DynamicGraphConvolution(256, 256, 10,device,stat_adj_mat[1],component=True)
        self.gcn_t = DynamicGraphConvolution(256, 256, 15,device,stat_adj_mat[2],component=True)
        self.gcn_triplet_inter = DynamicGraphConvolution(256, 256, 100,device,stat_adj_mat[3])

        self.head_i=Instrument_Head(256,num_instruments)
        self.head_v=Verb_Head(256,num_verbs)
        self.head_t=Target_Head(256,num_targets)
        self.head_triplet=Triplet_Head(256,num_triplets)
        self.device=device

    def forward(self,img,cam_i=None,cam_v=None,cam_t=None,cam_trip=None,second_stage_flag=False):
      with torch.cuda.amp.autocast(enabled=True):

        if second_stage_flag:
            features=self.backbone(img)

            mixed_input=torch.cat((img,cam_i,cam_v,cam_t,cam_trip),dim=1)
            mixed_features=self.cam_input_mixer(mixed_input)
            mixed_features=self.backbone_replica(mixed_features)

            features=features*mixed_features#self.backbone(mixed_features)#

            maps_i,out1_i,feats_i=self.sam_i(features)
            maps_t,out1_t,feats_t=self.sam_t(features,feats_i)
            maps_v,out1_v,feats_v=self.sam_v(features,feats_i,feats_t)



            # print(maps_i.shape,maps_v.shape,maps_t.shape,maps_triplet.shape) 

            node_features=torch.cat((maps_i,maps_v,maps_t),dim=-1)


            cams_concat=torch.cat((feats_i,feats_v,feats_t),dim=1)
            trip_cams=torch.einsum('bchw,cd->bdhw', cams_concat, self.stat_adj_mat[3].to(self.device))#, self.stat_adj_mat[4].to(device))
            trip_cams_size=trip_cams.shape
            trip_cams = trip_cams.view(trip_cams.size(0), trip_cams.size(1), -1) 
            out_3_trip=trip_cams.topk(1, dim=-1)[0].mean(dim=-1)
            trip_cams = torch.sigmoid(trip_cams)
            trip_cams=trip_cams.view(trip_cams_size[0],trip_cams_size[1],trip_cams_size[2],trip_cams_size[3])
            x_igcn=self.gcn_i(maps_i)
            x_vgcn=self.gcn_v(maps_v)
            x_tgcn=self.gcn_t(maps_t)

            tf_z_i=maps_i+x_igcn
            tf_z_v=maps_v+x_vgcn
            tf_z_t=maps_t+x_tgcn


            tf_z_inter=self.gcn_triplet_inter(node_features,residue=False)


            instrument_preds=self.head_i(tf_z_i,out1_i)
            verb_preds=self.head_v(tf_z_v,out1_v)
            target_preds=self.head_t(tf_z_t,out1_t)
            triplet_preds=self.head_triplet(tf_z_inter,out_3_trip)


            return triplet_preds,instrument_preds,verb_preds,target_preds#,feats_i,feats_v,feats_t







        else:
            features=self.backbone(img)

            maps_i,out1_i,feats_i=self.sam_i(features)
            maps_t,out1_t,feats_t=self.sam_t(features,feats_i)
            maps_v,out1_v,feats_v=self.sam_v(features,feats_i,feats_t)




            node_features=torch.cat((maps_i,maps_v,maps_t),dim=-1)



            cams_concat=torch.cat((feats_i,feats_v,feats_t),dim=1)
            trip_cams=torch.einsum('bchw,cd->bdhw', cams_concat, self.stat_adj_mat[3].to(self.device))
            trip_cams_size=trip_cams.shape
            trip_cams = trip_cams.view(trip_cams.size(0), trip_cams.size(1), -1) 
            out_3_trip=trip_cams.topk(1, dim=-1)[0].mean(dim=-1)
            trip_cams = torch.sigmoid(trip_cams)
            trip_cams=trip_cams.view(trip_cams_size[0],trip_cams_size[1],trip_cams_size[2],trip_cams_size[3])

            x_igcn=self.gcn_i(maps_i)
            x_vgcn=self.gcn_v(maps_v)
            x_tgcn=self.gcn_t(maps_t)

            tf_z_i=maps_i+x_igcn
            tf_z_v=maps_v+x_vgcn
            tf_z_t=maps_t+x_tgcn

            tf_z_inter=self.gcn_triplet_inter(node_features,residue=False)


            instrument_preds=self.head_i(tf_z_i,out1_i)
            verb_preds=self.head_v(tf_z_v,out1_v)
            target_preds=self.head_t(tf_z_t,out1_t)
            triplet_preds=self.head_triplet(tf_z_inter,out_3_trip)

            return triplet_preds,instrument_preds,verb_preds,target_preds,feats_i,feats_v,feats_t,trip_cams#,dy_adj,dy_adj_i,dy_adj_v,dy_adj_t



def create_backbone(config,device):

    dino_path=config["dino_path"]
    import sys
    sys.path.insert(0,dino_path)#'/home/raviteja/Cholec_Triplet/kv3_exps/dino/')

    import utils
    from vision_transformer import DINOHead

    
    model_sd_path=config["model_sd_path"]#'/home/raviteja/Cholec_Triplet/kv3_exps/dino_test_output/r50/checkpoint.pth'

    torch.manual_seed(7)

    teacher =models.__dict__['resnet50']()
    embed_dim = teacher.fc.weight.shape[1]
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, 65536, False))
    model=teacher

    dict_teacher=torch.load(model_sd_path,map_location=device)['teacher']
    new_teacher_dict={}
    for k in list(teacher.state_dict().keys()):
        new_teacher_dict[k]=dict_teacher['module.'+k]
    model.load_state_dict(new_teacher_dict)

    
    model1=nn.Sequential(*(list(model.children())[:-2]))

    tmp1=list(model.children())[0]
    
    backbone=nn.Sequential(*(list(tmp1.children())[:-3]))

    return backbone

def get_model(c_i,c_v,c_t,c_ivt,adj_mat,config,device):
    
    torch.manual_seed(7)
    backbone=create_backbone(config,device)
    model=Triplet_ADD_GCN(backbone,c_i,c_v,c_t,c_ivt,device,adj_mat)
    return model
