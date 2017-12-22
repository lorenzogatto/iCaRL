# iCaRL: incremental Class and Representation Learning

## Applied on ImageNet using Tensorflow and CaffeNet/VGG

##### Requirements
- Tensorflow (version 1.1)
- Scipy (working with 0.19)
- Core50 dataset
- Pretrained networks for caffenet and vggnet-m (http://www.robots.ox.ac.uk/~vgg/research/deep_eval/) 
obtained with https://github.com/ethereon/caffe-tensorflow.

#### Training

##### Launching the code
Execute ``main_icarl_core50.py`` to launch the code. 
Settings can easily be changed by passing parameters to the function. Call the script with ``-h`` for infos.
Some parameters, like dataset location, are hardcoded in that file.

Example execution command: ``python2.7 main_midvggm_tf.py --network mvggnet --image_size 128 --run NC_inc/run0 --lr_1 0.02 --lr_o 0.05 --stored_images 10 > output.txt``

##### Output files
- ``model-iteration-i.pickle``: storing of the network parameters after training each increment of classes
- ``Xclass_means.pickle`` : 4D tensor. Xclass_means[:,i,0,j] corresponds to the mean-of-examplars of the class i after the j-th increment of classes. Xclass_means[:,i,1,j] corresponds to the corresponding the theoretical class mean used by the NCM.
- ``Xfiles_protoset.pickle`` : list with Xfiles_protoset[i] containing the ranked exemplars for the class i. At each increment of classes, a smaller subset of the exemplars of the previous classes is used as the memory is fixed.

#### Testing

##### Launching the code
As we save the classifier, exemplars and weights after each increment, we can evaluate and compare the performances after each increment of classes. Execute ``test_icarl_core50.py`` if you want the cumulative performances after each increment.
Run the script with the same parameters you used for training.

##### Output file
- ``results_topX_acc_Y_clZ.npy``: accuracy file with each line corresponding to an increment. 1st column is with iCaRL, 2nd column with Hybrid 1 and 3rd column is the theoretical case of NCM.

####Parameters I used
Always 5 epochs and 50 img/classe unless differently specified.

CaffeNet: --lr_1: 0.08 --lr_o 0.04

MVGGNet: --lr_1: 0.02 --lr_o 0.05

To run the networks in full size, you'd need to modify
``mvggm.py`` or ``mcaffenet.py`` with
``net.load(model_data_path, sess, skip_op=['fc8'])`` to make it skip 
(from the loading of the pretrained model) only the last layer, 
changing the names 'mid_fc*' to 'fc*' in the network specification
and changing the dimensions of the first 2 FC layers to 4096. 