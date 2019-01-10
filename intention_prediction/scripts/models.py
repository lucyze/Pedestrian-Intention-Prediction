import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import ReLU

def make_mlp(dim_list, activations, batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out, activation in zip(dim_list[:-1], dim_list[1:], activations):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
            
    return nn.Sequential(*layers)

def convolutionize(layers, input_size):
    l = []
    x = Variable(torch.zeros(torch.Size((1,) + input_size)))

    for m in layers:
        if isinstance(m, nn.Linear):
            n = nn.Conv2d(
                    in_channels = x.size(1), 
                    out_channels = m.weight.size(0), 
                    kernel_size = (x.size(2), x.size(3)))
            n.weight.data.view(-1).copy_(m.weight.data.view(-1))
            n.bias.data.view(-1).copy_(m.bias.data.view(-1))

        l.append(m)
        x = m(x)

# ======= CNN LSTM MODEL =========== #
class CNNLSTM1(nn.Module):
    def __init__(
        self, embedding_dim=256, h_dim=32, mlp_dim=64, dropout=0.0, grad=False
    ):
        super(CNNLSTM1, self).__init__()

        # parameters
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # gradients
        self.gradients = None

        # CNN Feature Extractor
        self.model = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[0])

        # feature embedder
        self.feature_embedder = nn.Linear(1536, embedding_dim)

        # LSTM
        self.lstm = nn.LSTM(
                embedding_dim, h_dim, 1, batch_first=False
        )

        # Linear classifier
        self.linear_classifier = nn.Linear(h_dim, 2)

        # CNN gradients disabled
        if(grad==False):        
                for i,(name, param) in enumerate(self.model.named_parameters()):
                    if param.requires_grad: #and i != 24 or i!= 25:
                        #print(i,name,param.size())
                        param.requires_grad = False  
        # CNN gradients enabled for guided backprop        
        else: 
                for name, param in self.model.named_parameters():         
                    if param.requires_grad:
                        param.requires_grad = True       
                self.hook_layers()
                self.update_relus()

    # Set hook to the first CNN layer
    def hook_layers(self):    
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            # grad_out[0].size = timesteps x 64 x 100 x 40 
            # grad_in[0].size = None
            # grad_in[1].size = 64 x 3 x 3 x 3 (64 3x3x3 filters)
            # grad_in[2].size = 64 (biases)
        self.model[0].register_backward_hook(hook_function)

    # Updates relu activation functions so that it only returns positive gradients
    def update_relus(self):
        # Clamp negative gradients to 0
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through convolutional feature extractor and hook up ReLUs with relu_hook_function
        for i in range(len(self.model)):
            if isinstance(self.model[i], ReLU):
                self.model[i].register_backward_hook(relu_hook_function)
        ## Loop through MLP and hook up ReLUs with relu_hook_function
        #for i in range(len(self.classifier)):
        #    if isinstance(self.classifier[i], ReLU):
        #        self.classifier[i].register_backward_hook(relu_hook_function)

    def init_hidden(self, batch):
        if(torch.cuda.is_available()):
          return(
          torch.zeros(1, batch, self.h_dim).cuda(),
          torch.zeros(1, batch, self.h_dim).cuda()
          )
        else:
          return(
            torch.zeros(1, batch, self.h_dim),
            torch.zeros(1, batch, self.h_dim)
            )

    def forward(self, images_pedestrian_all, input_as_var = False, classify_every_timestep=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        # batch = number of pedestrians where sequence length of each pedestrian varies
        batch = len(images_pedestrian_all)

        # We are not going to classify at every timestep.
        if(classify_every_timestep==False):
                # for each pedestrian
                #features_pedestrian_all = []
                state_all = []
                for images_pedestrian_i in images_pedestrian_all:
                           
                        # sequence length
                        seq_len = images_pedestrian_i.size(0)
                                  
                        # if we want the input to be a Variable
                        # used for guided backprop
                        if(input_as_var == True):
                            if(torch.cuda.is_available()):
                              images_pedestrian_i = Variable(images_pedestrian_i.cuda(), requires_grad=True)
                            else:
                              images_pedestrian_i = Variable(images_pedestrian_i, requires_grad=True)
                        else:
                            if(torch.cuda.is_available()):
                              images_pedestrian_i = images_pedestrian_i.cuda()
                            else:
                              images_pedestrian_i = images_pedestrian_i                              

                        # send all the images of the current pedestrian through the CNN feature extractor
                        features_pedestrian_i = self.model(images_pedestrian_i)
                        features_pedestrian_i = features_pedestrian_i.view(seq_len, -1)

                        # embed the features
                        features_pedestrian_i = self.feature_embedder(features_pedestrian_i) # (seq length, 64)
                        features_pedestrian_i = F.dropout(F.relu(features_pedestrian_i), p=self.dropout)
                        features_pedestrian_i = torch.unsqueeze(features_pedestrian_i, 1)
                        
                        # send through lstm and get the final output
                        state_tuple = self.init_hidden(1)
                        output, state = self.lstm(features_pedestrian_i)       
                        state_all.append(F.relu(F.dropout(state[0].squeeze(), p=self.dropout)))

                state_all = torch.stack(state_all, dim=0)
                y_pred = self.linear_classifier(state_all)
                return y_pred

        
        # We are going to classify at every timestep
        if(classify_every_timestep==True):
                y_pred_all = []
                # for each pedestrian
                for images_pedestrian_i in images_pedestrian_all:
                        y_pred_i = []                 

                        # sequence length
                        seq_len = images_pedestrian_i.size(0)
                        
                        # if we want the input to be a Variable
                        # used for guided backprop
                        if(input_as_var == True):
                            if(torch.cuda.is_available()):
                              images_pedestrian_i = Variable(images_pedestrian_i.cuda(), requires_grad=True)
                            else:
                              images_pedestrian_i = Variable(images_pedestrian_i, requires_grad=True)                              
                        else:
                            if(torch.cuda.is_available()):
                              images_pedestrian_i = images_pedestrian_i.cuda()
                            else:
                              images_pedestrian_i = images_pedestrian_i

                        # send all the images of the current pedestrian through the CNN feature extractor
                        features_pedestrian_i = self.model(images_pedestrian_i)
                        features_pedestrian_i = features_pedestrian_i.view(seq_len, -1)

                        # embed the features
                        features_pedestrian_i = self.feature_embedder(features_pedestrian_i) # (seq length, 64)
                        
                        # send through lstm and classify at each timestep
                        state_tuple = self.init_hidden(1)
                        for f in features_pedestrian_i:
                             output, state_tuple = self.lstm(f.view(1,1,-1), state_tuple)      
                             y_pred = self.linear_classifier(state_tuple[0])
                             y_pred_i.append(y_pred.squeeze().max(0)[1])
                        # append classification results for current pedestrian
                        y_pred_all.append(torch.stack(y_pred_i, dim=0))
                return y_pred_all

# ======= CNN LSTM MODEL =========== #
class CNNLSTM2(nn.Module):
    def __init__(
        self, embedding_dim=256, h_dim=32, mlp_dim=64, dropout=0.0, grad=False
    ):
        super(CNNLSTM2, self).__init__()

        # parameters
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # gradients
        self.gradients = None

        # CNN Feature Extractor
        self.model = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[0])
        self.model_relu = nn.ReLU()
        self.model_dropout = nn.Dropout(p=self.dropout)

        # LSTM
        self.lstm = nn.LSTM(1536, h_dim, 1, batch_first=False)
        self.lstm_relu = nn.ReLU()
        self.lstm_dropout = nn.Dropout(p=self.dropout)

        # Linear classifier
        self.linear_classifier = nn.Linear(h_dim, 2)

        # CNN gradients disabled
        if(grad==False):        
            for i,(name, param) in enumerate(self.model.named_parameters()):
                if param.requires_grad:                        
                    param.requires_grad = False 
                        
        # CNN gradients enabled 
        # - currently enabled only for guided backprop        
        else:      
            self.hook_layers()
            self.update_relus()
                
    # Set hook to the first layer
    def hook_layers(self):  
        # grad_out[0].size = timesteps x 64 x 100 x 40 
        # grad_in[0].size = None
        # grad_in[1].size = 64 x 3 x 3 x 3 (64 3x3x3 filters)
        # grad_in[2].size = 64 (biases)
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        self.model[0].register_backward_hook(hook_function)

    # Updates relu activation functions so that it only returns positive gradients
    def update_relus(self):    
        
        # Clamp negative gradients to 0
        def relu_forward_hook_function(module, grad_in, grad_out):
            module.input_kept = grad_in[0]
            #return (torch.clamp(grad_in[0], min=0.0),) # can be replaced by RELU
            
        def relu_backward_hook_function(module, grad_in, grad_out):
            return F.relu(grad_out[0]) * F.relu(module.input_kept).sign(),
                    
        # Loop through convolutional feature extractor and hook up ReLUs with relu_hook_function
        for i in range(len(self.model)):
            if isinstance(self.model[i], ReLU):
                self.model[i].register_forward_hook(relu_forward_hook_function)
                self.model[i].register_backward_hook(relu_backward_hook_function)
        self.model_relu.register_forward_hook(relu_forward_hook_function)
        self.model_relu.register_backward_hook(relu_backward_hook_function)
        self.lstm_relu.register_forward_hook(relu_forward_hook_function)
        self.lstm_relu.register_backward_hook(relu_backward_hook_function)

    def init_hidden(self, batch):
        if(torch.cuda.is_available()):
            return(
                torch.zeros(1, batch, self.h_dim).cuda(),
                torch.zeros(1, batch, self.h_dim).cuda()
            )
        else:
            return(
                torch.zeros(1, batch, self.h_dim),
                torch.zeros(1, batch, self.h_dim)
          )          

    def forward(self, images_pedestrian_all, input_as_var = False, classify_every_timestep=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        # batch = number of pedestrians where sequence length of each pedestrian varies
        batch = len(images_pedestrian_all)

        if(classify_every_timestep==False):
                # for each pedestrian
                #features_pedestrian_all = []
                state_all = []
                for images_pedestrian_i in images_pedestrian_all:
                           
                        # sequence length
                        seq_len = images_pedestrian_i.size(0)
                                  
                        # if we want the input to be a Variable
                        # used for guided backprop
                        if(input_as_var == True):
                            if(torch.cuda.is_available()):
                                images_pedestrian_i = Variable(images_pedestrian_i.cuda(), requires_grad=True)
                            else:
                                images_pedestrian_i = Variable(images_pedestrian_i, requires_grad=True)                              
                        else:
                            if(torch.cuda.is_available()):
                                images_pedestrian_i = images_pedestrian_i.cuda()
                            else:
                                images_pedestrian_i = images_pedestrian_i

                        # send all the images of the current pedestrian through the CNN feature extractor
                        # relu-dropout-flatten
                        features_pedestrian_i = self.model(images_pedestrian_i)
                        features_pedestrian_i = features_pedestrian_i.view(seq_len, -1)       
                        features_pedestrian_i = self.model_dropout(self.model_relu(features_pedestrian_i))
                        
                        # send through lstm and get the final output
                        features_pedestrian_i = torch.unsqueeze(features_pedestrian_i, 1)
                        state_tuple = self.init_hidden(1)
                        output, state = self.lstm(features_pedestrian_i)       
                        state_all.append(self.lstm_dropout(self.lstm_relu(state[0].squeeze())))

                state_all = torch.stack(state_all, dim=0)
                y_pred = self.linear_classifier(state_all)
                return y_pred

        if(classify_every_timestep==True):
                y_pred_all = []
                # for each pedestrian
                for images_pedestrian_i in images_pedestrian_all:
                        y_pred_i = []                 

                        # sequence length
                        seq_len = images_pedestrian_i.size(0)
                        
                        # if we want the input to be a Variable
                        # used for guided backprop
                        if(input_as_var == True):
                            if(torch.cuda.is_available()):
                                images_pedestrian_i = Variable(images_pedestrian_i.cuda(), requires_grad=True)
                            else:
                                images_pedestrian_i = Variable(images_pedestrian_i, requires_grad=True)                              
                        else:
                            if(torch.cuda.is_available()):
                                images_pedestrian_i = images_pedestrian_i.cuda()
                            else:
                                images_pedestrian_i = images_pedestrian_i

                        # send all the images of the current pedestrian through the CNN feature extractor
                        features_pedestrian_i = self.model(images_pedestrian_i)
                        features_pedestrian_i = features_pedestrian_i.view(seq_len, -1)

                        # relu-dropout-flatten
                        features_pedestrian_i = F.dropout(F.relu(features_pedestrian_i), p=self.dropout)
                        features_pedestrian_i = torch.unsqueeze(features_pedestrian_i, 1)
                        
                        # send through lstm and classify at each timestep
                        state_tuple = self.init_hidden(1)
                        for f in features_pedestrian_i:
                            # get the state
                            output, state_tuple = self.lstm(f.view(1,1,-1), state_tuple)  
                            # relu-dropout-classify
                            y_pred = self.linear_classifier(F.relu(F.dropout(state_tuple[0])))
                            y_pred_i.append(y_pred.squeeze().max(0)[1]) #y_pred_i.append(y_pred.squeeze().max(0)[1])                       
                        # append classification results for current pedestrian
                        y_pred_all.append(torch.stack(y_pred_i, dim=0))
                return y_pred_all

# ======= CNN LSTM MODEL =========== #
class CNNLSTM3(nn.Module):
    def __init__(
        self, embedding_dim=256, h_dim=32, mlp_dim=64, dropout=0.0, grad=False
    ):
        super(CNNLSTM3, self).__init__()

        # parameters
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # gradients
        self.gradients = None

        # CNN Feature Extractor
        self.model = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[0])
        self.model_relu = nn.ReLU()
        self.model_dropout = nn.Dropout(p=self.dropout)

        # LSTM
        self.lstm = nn.LSTM(1536, h_dim, 1, batch_first=False)
        self.lstm_relu = nn.ReLU()
        self.lstm_dropout = nn.Dropout(p=self.dropout)

        # Linear classifier
        self.linear_classifier = nn.Linear(h_dim, 2)

        # CNN gradients disabled
        if(grad==False):        
            for i,(name, param) in enumerate(self.model.named_parameters()):
                if param.requires_grad:                        
                    param.requires_grad = False 
                        
        # CNN gradients enabled 
        # - currently enabled only for guided backprop        
        else:      
            self.hook_layers()
            self.update_relus()
                
    # Set hook to the first layer
    def hook_layers(self):  
        # grad_out[0].size = timesteps x 64 x 100 x 40 
        # grad_in[0].size = None
        # grad_in[1].size = 64 x 3 x 3 x 3 (64 3x3x3 filters)
        # grad_in[2].size = 64 (biases)
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        self.model[0].register_backward_hook(hook_function)

    # Updates relu activation functions so that it only returns positive gradients
    def update_relus(self):    
        
        # Clamp negative gradients to 0
        def relu_forward_hook_function(module, grad_in, grad_out):
            module.input_kept = grad_in[0]
            #return (torch.clamp(grad_in[0], min=0.0),) # can be replaced by RELU
            
        def relu_backward_hook_function(module, grad_in, grad_out):
            return F.relu(grad_out[0]) * F.relu(module.input_kept).sign(),
                    
        # Loop through convolutional feature extractor and hook up ReLUs with relu_hook_function
        for i in range(len(self.model)):
            if isinstance(self.model[i], ReLU):
                self.model[i].register_forward_hook(relu_forward_hook_function)
                self.model[i].register_backward_hook(relu_backward_hook_function)
        self.model_relu.register_forward_hook(relu_forward_hook_function)
        self.model_relu.register_backward_hook(relu_backward_hook_function)
        self.lstm_relu.register_forward_hook(relu_forward_hook_function)
        self.lstm_relu.register_backward_hook(relu_backward_hook_function)

    def init_hidden(self, batch):
        if(torch.cuda.is_available()):
            return(
                torch.zeros(1, batch, self.h_dim).cuda(),
                torch.zeros(1, batch, self.h_dim).cuda()
            )
        else:
            return(
                torch.zeros(1, batch, self.h_dim),
                torch.zeros(1, batch, self.h_dim)
          )          

    def forward(self, images_pedestrian_all, input_as_var = False, classify_every_timestep=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        # batch = number of pedestrians where sequence length of each pedestrian varies
        batch = len(images_pedestrian_all)

        if(classify_every_timestep==False):
                # for each pedestrian
                #features_pedestrian_all = []
                state_all = []
                for images_pedestrian_i in images_pedestrian_all:
                           
                        # sequence length
                        seq_len = images_pedestrian_i.size(0)
                                  
                        # if we want the input to be a Variable
                        # used for guided backprop
                        if(input_as_var == True):
                            if(torch.cuda.is_available()):
                                images_pedestrian_i = Variable(images_pedestrian_i.cuda(), requires_grad=True)
                            else:
                                images_pedestrian_i = Variable(images_pedestrian_i, requires_grad=True)                              
                        else:
                            if(torch.cuda.is_available()):
                                images_pedestrian_i = images_pedestrian_i.cuda()
                            else:
                                images_pedestrian_i = images_pedestrian_i

                        # send all the images of the current pedestrian through the CNN feature extractor
                        # relu-dropout-flatten
                        features_pedestrian_i = self.model(images_pedestrian_i)
                        features_pedestrian_i = features_pedestrian_i.view(seq_len, -1)       
                        features_pedestrian_i = self.model_dropout(self.model_relu(features_pedestrian_i))
                        
                        # send through lstm and get the final output
                        features_pedestrian_i = torch.unsqueeze(features_pedestrian_i, 1)
                        state_tuple = self.init_hidden(1)
                        output, state = self.lstm(features_pedestrian_i)       
                        state_all.append(self.lstm_dropout(self.lstm_relu(state[0].squeeze())))
                        
                        # send through lstm and classify at each timestep
                        state_tuple = self.init_hidden(1)
                        for f in features_pedestrian_i:
                            # get the state
                            output, state_tuple = self.lstm(f.view(1,1,-1), state_tuple)  
                            # relu-dropout-classify
                            y_pred = self.linear_classifier(F.relu(F.dropout(state_tuple[0])))
                            #print(y_pred, y_pred.squeeze().max(0)[1])
                            if(y_pred.squeeze().max(0)[1].data == 1):
                                return y_pred

                #input()
                #state_all = torch.stack(state_all, dim=0)
                #y_pred = self.linear_classifier(state_all)
                #return y_pred

        if(classify_every_timestep==True):
                y_pred_all = []
                # for each pedestrian
                for images_pedestrian_i in images_pedestrian_all:
                        y_pred_i = []                 

                        # sequence length
                        seq_len = images_pedestrian_i.size(0)
                        
                        # if we want the input to be a Variable
                        # used for guided backprop
                        if(input_as_var == True):
                            if(torch.cuda.is_available()):
                                images_pedestrian_i = Variable(images_pedestrian_i.cuda(), requires_grad=True)
                            else:
                                images_pedestrian_i = Variable(images_pedestrian_i, requires_grad=True)                              
                        else:
                            if(torch.cuda.is_available()):
                                images_pedestrian_i = images_pedestrian_i.cuda()
                            else:
                                images_pedestrian_i = images_pedestrian_i

                        # send all the images of the current pedestrian through the CNN feature extractor
                        features_pedestrian_i = self.model(images_pedestrian_i)
                        features_pedestrian_i = features_pedestrian_i.view(seq_len, -1)

                        # relu-dropout-flatten
                        features_pedestrian_i = F.dropout(F.relu(features_pedestrian_i), p=self.dropout)
                        features_pedestrian_i = torch.unsqueeze(features_pedestrian_i, 1)
                        
                        # send through lstm and classify at each timestep
                        state_tuple = self.init_hidden(1)
                        for f in features_pedestrian_i:
                            # get the state
                            output, state_tuple = self.lstm(f.view(1,1,-1), state_tuple)  
                            # relu-dropout-classify
                            y_pred = self.linear_classifier(F.relu(F.dropout(state_tuple[0])))
                            y_pred_i.append(y_pred.squeeze().max(0)[1]) #y_pred_i.append(y_pred.squeeze().max(0)[1])                       
                        # append classification results for current pedestrian
                        y_pred_all.append(torch.stack(y_pred_i, dim=0))
                return y_pred_all
            
# ======= CNN LSTM MODEL =========== #        
class CNNLSTMJAAD2(nn.Module):
    def __init__(
        self, embedding_dim=256, h_dim=32, mlp_dim=64, dropout=0.0, grad=False
    ):
        super(CNNLSTMJAAD2, self).__init__()

        # parameters
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # gradients
        self.gradients = None

        # CNN Feature Extractor
        self.model = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[0])
        self.model_relu = nn.ReLU()
        self.model_dropout = nn.Dropout(p=self.dropout)

        # LSTM
        self.lstm = nn.LSTM(
                1536, h_dim, 1, batch_first=False
        )

        # Linear classifier
        self.standing = nn.Linear(h_dim, 2)
        self.looking  = nn.Linear(h_dim, 2)
        self.walking  = nn.Linear(h_dim, 2)
        self.crossing = nn.Linear(h_dim, 2)

        # CNN gradients disabled
        if(grad==False):        
                for i,(name, param) in enumerate(self.model.named_parameters()):
                    if param.requires_grad: 
                        param.requires_grad = False  
        # CNN gradients enabled for guided backprop        
        else: 
                for name, param in self.model.named_parameters():         
                    if param.requires_grad:
                        param.requires_grad = True       
                self.hook_layers()
                self.update_relus()

    # Set hook to the first CNN layer
    def hook_layers(self):    
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            # grad_out[0].size = timesteps x 64 x 100 x 40 
            # grad_in[0].size = None
            # grad_in[1].size = 64 x 3 x 3 x 3 (64 3x3x3 filters)
            # grad_in[2].size = 64 (biases)
        self.model[0].register_backward_hook(hook_function)

    # Updates relu activation functions so that it only returns positive gradients
    def update_relus(self):
        # Clamp negative gradients to 0
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through convolutional feature extractor and hook up ReLUs with relu_hook_function
        for i in range(len(self.model)):
            if isinstance(self.model[i], ReLU):
                self.model[i].register_backward_hook(relu_hook_function)
        ## Loop through MLP and hook up ReLUs with relu_hook_function
        #for i in range(len(self.classifier)):
        #    if isinstance(self.classifier[i], ReLU):
        #        self.classifier[i].register_backward_hook(relu_hook_function)

    def init_hidden(self, batch):
        if(torch.cuda.is_available()):
            return(
                torch.zeros(1, batch, self.h_dim).cuda(),
                torch.zeros(1, batch, self.h_dim).cuda()
            )
        else:
            return(
                torch.zeros(1, batch, self.h_dim),
                torch.zeros(1, batch, self.h_dim)
            )

    def forward(self, images_pedestrian_all, input_as_var = False, classify_every_timestep=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        # batch = number of pedestrians where sequence length of each pedestrian varies
        batch = len(images_pedestrian_all)

        standing_pred_all = []
        looking_pred_all = []
        walking_pred_all = []
        crossing_pred_all = []
        
        # for each pedestrian
        for images_pedestrian_i in images_pedestrian_all:
        
                standing_pred_i = [] 
                looking_pred_i = []
                walking_pred_i = []
                crossing_pred_i = []
                      
                # sequence length
                seq_len = images_pedestrian_i.size(0)
                
                # if we want the input to be a Variable for guided backprop
                if(input_as_var == True):
                    if(torch.cuda.is_available()):
                        images_pedestrian_i = Variable(images_pedestrian_i.cuda(), requires_grad=True)
                    else:
                        images_pedestrian_i = Variable(images_pedestrian_i, requires_grad=True)                              
                else:
                    if(torch.cuda.is_available()):
                        images_pedestrian_i = images_pedestrian_i.cuda()
                    else:
                        images_pedestrian_i = images_pedestrian_i

                # send all the images of the current pedestrian through the CNN feature extractor
                features_pedestrian_i = self.model(images_pedestrian_i)
                features_pedestrian_i = features_pedestrian_i.view(seq_len, -1)
                features_pedestrian_i = self.model_dropout(self.model_relu(features_pedestrian_i))                   
                
                # send through lstm and classify at each timestep
                # - relu-dropout the states before classification
                # - but do not relu-dropout the states to the next timestep
                state_tuple = self.init_hidden(1)
                for f in features_pedestrian_i:
                    # get state
                    output, state_tuple = self.lstm(f.view(1,1,-1), state_tuple) 
                    
                    # dropout-relu classify action    
                    standing_pred = self.standing(F.relu(F.dropout(state_tuple[0])))     
                    looking_pred  = self.looking(F.relu(F.dropout(state_tuple[0])))                                
                    walking_pred  = self.walking(F.relu(F.dropout(state_tuple[0])))  
                    crossing_pred = self.crossing(F.relu(F.dropout(state_tuple[0])))
                                          
                    # list of length timestep
                    standing_pred_i.append(standing_pred.squeeze())
                    looking_pred_i.append(standing_pred.squeeze())
                    walking_pred_i.append(standing_pred.squeeze())
                    crossing_pred_i.append(standing_pred.squeeze())
                                                            
                # append classification results
                # dropout-relu classify decision
                standing_pred_all.append(torch.stack(standing_pred_i))
                looking_pred_all.append(torch.stack(looking_pred_i))
                walking_pred_all.append(torch.stack(walking_pred_i))
                crossing_pred_all.append(torch.stack(crossing_pred_i))              
        
        return standing_pred_all, looking_pred_all, walking_pred_all, crossing_pred_all                       
        #return torch.cat(standing_pred_all,dim=0), torch.cat(looking_pred_all,dim=0), torch.cat(walking_pred_all,dim=0), torch.cat(crossing_pred_all,dim=0)
