# Deep Learning Notebooks

All notebooks can work in google colab.(Change Runtime Type to GPU)

1. Pytorch basic and deeping learning training and evaluating(XOR classify as an example)

2. Classifing FashionMNIST dataset images(as 1D tensor) using mulit-Linear layer network. Training the network and saving the trained model with best validation accuracy to files. Testing the network by loading the model from that file. In this example, we train the network with different activation functions(sigmoid, leakyReLU). This training function can alse be used to do other hyperparamter grid-search.

3. Add more statistic(train_loss ,train_acc, val_acc, test_acc) and save them to json file(result.json). Add plot. Exploring initialization methods( xavier initialization, kaiming initialization) and optimization methods(SGD, SGD momentum, Adam) by hand.
