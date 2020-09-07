


class Encoder_gaussian(torch.nn.Module):
    def __init__(self, param):
        super(Encoder_gaussian, self).__init__()
        self.input_nc = param['input_nc']
        self.output_nc = param['output_nc']
        self.ngf = param['ngf']
        self.gpu_ids = param['gpu_ids']
        self.filt_h = param['filt_h']
        self.filt_w = param['filt_w']
        self.batch_size = param['batch_size']
        self.padding = param['padding']
        self.win_length = param['win_length']
        self.patch_length = param['patch_length']
        self.len_after_max_pool = int((self.win_length - param['pool_window'])/param['pool_stride']) + 1

        self.num_nodes_fnn_mod = param['num_nodes_fnn_mod']
        self.num_nodes_mean_var_mod = param['num_nodes_mean_var_mod']
        self.ngf_mod = param['ngf_mod']
        self.filt_h_mod = param['filt_h_mod']
        self.filt_w_mod = param['filt_w_mod']
        self.fc = None
        
        self.means = torch.nn.Parameter((torch.rand(self.ngf)*(-5) + 1.3).float().cuda())
        t = range(-self.filt_h/2, self.filt_h/2)
        self.temp = torch.from_numpy((np.reshape(t, [self.filt_h, ]))).float().cuda() + 1
        
        self.max_pool_layer = torch.nn.MaxPool2d((param['pool_window'], self.filt_w), stride=(param['pool_stride'], 1), return_indices=True)
        self.avg_pool_layer = torch.nn.AvgPool2d((self.len_after_max_pool, self.filt_w), stride=(1, 1))
        
        self.r1 = torch.nn.Parameter((torch.rand(1, 1, self.filt_h_mod, 1).float().cuda())
        self.r2 = torch.nn.Parameter((torch.rand(1, 1, self.filt_h_mod, 1).float().cuda())
        self.s1 = torch.nn.Parameter((torch.rand(1, 1, self.filt_w_mod, 1).float().cuda())
        self.s2 = torch.nn.Parameter((torch.rand(1, 1, self.filt_w_mod, 1).float().cuda())
        
        self.fnn_layer1 = torch.nn.Linear(param['num_nodes_fnn_mod'][0], param['num_nodes_fnn_mod'][1])
        # self.fnn_layer2 = torch.nn.Linear(param['num_nodes_fnn_mod'][1], param['num_nodes_fnn_mod'][2])

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        # Updating 1-D Gaussian kernels (acoustic FB) using updated Means Parameter
        means_sorted = torch.sort(self.means)[0]
        kernels = (torch.zeros([self.ngf, self.filt_h]).cuda())
        for i in range(self.ngf):
            kernels[i, :] = torch.cos(np.pi * torch.sigmoid(means_sorted[i]) * self.temp) * torch.exp(- (((self.temp)**2)/(2*(((1/(torch.sigmoid(means_sorted[i])+1e-3))*10)**2 + 1e-5))))

        kernels = (torch.reshape(kernels, (kernels.shape[0], 1, kernels.shape[1], 1)))
        
        # Input is [B, C=1, H=400, W=120] with patch_length=120
        x = x.permute(0,3,2,1) # B,W,H,C=1
        x = x.contiguous().view(self.batch_size*self.patch_length, 1, -1, 1) # BW,1,H,C=1

        out = F.conv2d(x, kernels, padding=(self.padding, 0))  # BW,C=ngf,H,1
        out = out.contiguous().view(self.batch_size, self.patch_length, self.ngf, -1).permute(0,2,3,1)   # B,W,ngf,H --> B,ngf,H,W
        out, indices = (self.max_pool_layer(out))  #B,C=ngf,(H-poolW)/poolS+1,W
        out = torch.sigmoid(out)
        spec = torch.log(self.avg_pool_layer(out))  # B,C=ngf,H=1,W
        
        out2 = spec.contiguous().view(self.batch_size*self.ngf, 1, self.patch_length, -1)  # B*ngf,C=1,W=patch_len, 1
        
        # RATE filtering
        out3 = F.conv2d(out2, self.r1, padding=(2, 0))  # First rate filtering
        out4 = self.tanh(F.conv2d(self.tanh(out2-out3), self.r2, padding=(2, 0))+out3)  # Second rate filtering over residual, adding the 2 feature maps afterwards
        out4 = out4.contiguous().view(self.batch_size, self.ngf, -1, 1)
        out4 = out4.permute(0,2,1,3).contiguous().view(self.batch_size*self.patch_length, 1, -1, 1)   # B*W, 1, ngf, 1
        
        # SCALE filtering
        out5 = F.conv2d(out4, self.s1, padding=(2, 0))  # First scale filtering
        out6 = self.tanh(F.conv2d(self.tanh(out4-out5), self.s2, padding=(2, 0))+out5)  # Second scale filtering over residual, adding the 2 feature maps afterwards
        out6 = out6.contiguous().view(self.batch_size,self.patch_length, -1, 1).permute(0,2,1,3)
        out = out6.contiguous().view(self.batch_size*self.ngf, -1)   # B*ngf, W=patch_len
        out = (self.tanh(self.fnn_layer1(out)))  # B*ngf, latent_dim
        
        return spec, out, indices
        
        
class Decoder(torch.nn.Module):
    def __init__(self, param):
        super(Decoder, self).__init__()
        self.input_nc = param['input_nc']
        self.output_nc = param['output_nc']
        self.ngf = param['ngf']
        self.gpu_ids = param['gpu_ids']
        self.filt_h = param['filt_h']
        self.filt_w = param['filt_w']
        self.batch_size = param['batch_size']
        self.padding = param['padding']
        self.win_length = param['win_length']
        self.patch_length = param['patch_length']
        self.fc = None
        self.len_after_max_pool = int((self.win_length - param['pool_window'])/param['pool_stride']) + 1

        self.num_nodes_fnn_mod = param['num_nodes_fnn_mod']
        self.num_nodes_mean_var_mod = param['num_nodes_mean_var_mod']
        self.ngf_mod = param['ngf_mod']
        self.filt_h_mod = param['filt_h_mod']
        self.filt_w_mod = param['filt_w_mod']

        self.fnn_layer3 = torch.nn.Linear(param['num_nodes_mean_var_mod'], param['num_nodes_fnn_mod'][2])
        self.fnn_layer4 = torch.nn.Linear(param['num_nodes_fnn_mod'][2], param['num_nodes_fnn_mod'][3]*self.len_after_max_pool)
        self.tanh = torch.nn.Tanh()
        self.max_unpool_layer = torch.nn.MaxUnpool2d((param['pool_window'], self.filt_w), stride=(param['pool_stride'], 1))
        self.output_layer = torch.nn.ConvTranspose2d(param['ngf'], self.output_nc, kernel_size=(self.filt_h, self.filt_w), padding=(self.padding, 0), bias=False)
        
    def forward(self, x, indices):
        # print(x.shape)  # B*ngf, latent
        out = (self.tanh(self.fnn_layer3(x))) # B*ngf, fnn_node
        out_spec = torch.exp(out).contiguous().view(self.batch_size, self.ngf, -1) # [B, ngf, patch_len]
        

        # Repmat for undoing avg pooling
        out = (out_spec).unsqueeze(3).repeat(1,1,1,self.len_after_max_pool).permute(0,1,3,2)  # # [B, ngf, patch_len] to # [B, ngf, patch_len, len_after_max_pool] to [B, ngf, len_after_max_pool, patch_len]
        out = self.max_unpool_layer(out, indices)  # [B, ngf, win_len, patch_len]
        out_deconv = out.permute(0,3,1,2).contiguous().view(self.batch_size*self.patch_length, self.ngf, -1, 1)  # [B*patch_len, ngf, patch_len, 1]
        out = (self.output_layer(out_deconv))  # [B*patch_len, 1, win_len, 1]
        out = out.contiguous().view(self.batch_size, self.patch_length, -1, 1).permute(0,3,2,1)

        return out_spec, out
        
        
class VAE(torch.nn.Module):

    def __init__(self, param):
        super(VAE, self).__init__()
        self.encoder = Encoder_gaussian(param)
        self.decoder = Decoder(param)
        self._enc_mu = torch.nn.Linear(param['num_nodes_fnn_mod'][1], param['num_nodes_mean_var_mod'])
        self._enc_log_sigma = torch.nn.Linear(param['num_nodes_fnn_mod'][1], param['num_nodes_mean_var_mod'])
        self.use_cuda = param['cuda']

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        if self.use_cuda:
            std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()
        else:
            std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma
        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, inp):
        E_conv_out, h_enc, indices = self.encoder(inp)
        z = self._sample_latent(h_enc)
        D_deconv_in, out = self.decoder(z, indices)
        return E_conv_out, h_enc, D_deconv_in, out.view(inp.shape)
                                                     
                                                     
        
