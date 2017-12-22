# iCaRL: incremental Class and Representation Learning

## Applied on ImageNet using Tensorflow and CaffeNet/VGG

##### Requirements
- Tensorflow (version 1.1)
- Scipy (working with 0.19)
- Core50 dataset

#### Training

##### Launching the code
Execute ``main_icarl_core50.py`` to launch the code. 
Settings can easily be changed by passing parameters to the function. Call the script with ``-h`` for infos.
Some parameters, like dataset location, are hardcoded in that file.

PS: if your data presents a different folder architecture, you can change it in ``utils_data.py``

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
