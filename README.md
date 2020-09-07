# Unsupervised representation learning from raw waveform using convolutional variational autoencoder (CVAE)

We propose a deep representation learning approach using the raw speech waveform in an unsupervised learning paradigm. The first layer of the proposed deep model performs acoustic filtering while the subsequent layer performs modulation filtering. The acoustic filterbank is implemented using cosine-modulated Gaussian filters whose parameters are learned. The modulation filtering is performed on log transformed outputs of the first layer and this is achieved using a skip-connection based architecture. The outputs from this two layer filtering are fed to the variational autoencoder model. All the model parameters including the filtering layers are learned using the VAE cost function. 
We later employ the learned representations (second layer outputs) in a speech recognition task. The ASR acoustic model code is not provided here.

******************************************************************

Script Generator.py consists of the network architecture.

******************************************************************

Reference paper:

P. Agrawal and S. Ganapathy, "Unsupervised Raw Waveform Representation Learning for ASR", INTERSPEECH, 2019.

******************************************************************

01-Sept-2019 See the file LICENSE for the licence associated with this software.
