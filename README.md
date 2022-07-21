# Deep Learning Notebooks

All notebooks can work in google colab.(Change Runtime Type to GPU)

1. Pytorch_basic. Pytorch basic and deeping learning training and evaluating(XOR classify as an example)

2. Activation. Classifing FashionMNIST dataset images(as 1D tensor) using mulit-Linear layer network. Training the network and saving the trained model with best validation accuracy to files. Testing the network by loading the model from that file. In this example, we train the network with different activation functions(sigmoid, leakyReLU). This training function can alse be used to do other hyperparamter grid-search.

3. Optimization. Add more statistic(train_loss ,train_acc, val_acc, test_acc) and save them to json file(result.json). Add plot. Exploring initialization methods( xavier initialization, kaiming initialization) and optimization methods(SGD, SGD momentum, Adam) by hand.

4. CNN. Training the model using pytorch lightning framework. Define lightning module(init param, configure_optimizer, training_step, validation_step, test_step). Train using trianer.fit(lightning, train_dataloader). Test using trainer.test(lightning, test_dataloader. Task: CIFAR10 image classification task. Network: GoogleNet(Inception block), ResNet, DenseNet.

5. Transformer. Scaled dot product attention and Multi-Head Attention Layer. The Transformer architecture is based on the Multi-Head Attention layer and applies multiple of them in a ResNet-like block(Layernorm). Explore positional encoding(Since transformer is permutation-equivariant and learning rate warm up(gradient). Usage example: Sequence-to-Sequence problem(reverse the sequence). Set Anomally Detection(no need for positional encoding).

6. RNN. Grpha Convolution Network(GCN) and Graph Attention Network(GAT). Tasks: Node-level task(node classification). Graph-level task(graph classfication).
