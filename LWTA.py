
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable

class LWTA(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
       

    def forward(self, inputs):
        Masks=[]
        for this_input in inputs:
            this_input=this_input.data
            shape = list(this_input.size())

            
            #the lwta code starts here
            ###############
            non_zer_size=(int)(shape[0]/self.pool_size)
            y=this_input.view(1,non_zer_size,(int)(self.pool_size))
            m=y.max(2,keepdim=True)
            row_index=torch.LongTensor(list(range(non_zer_size))).cuda()
            row_index=row_index*self.pool_size
            mm=m[1].squeeze()
            row_index=row_index+mm
    
            Mask=torch.zeros(shape).cuda()
            Mask[row_index]=1
            max_dim = len(shape) - 1
            
            Masks.append(Mask.unsqueeze(0))
            
       
        
        Masks=torch.cat(Masks)
        Masks = Variable(Masks, requires_grad=False)


        output =Masks*inputs
        return output