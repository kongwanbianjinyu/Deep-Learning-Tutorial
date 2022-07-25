# Deep Learning Notebooks

All notebooks can work in google colab.(Change Runtime Type to GPU)

1. Pytorch_basic. Pytorch basic and deeping learning training and evaluating(XOR classify as an example)

2. Activation. Classifing FashionMNIST dataset images(as 1D tensor) using mulit-Linear layer network. Training the network and saving the trained model with best validation accuracy to files. Testing the network by loading the model from that file. In this example, we train the network with different activation functions(sigmoid, leakyReLU). This training function can alse be used to do other hyperparamter grid-search.

3. Initialization_Optimization. Add more statistic(train_loss ,train_acc, val_acc, test_acc) and save them to json file(result.json). Add plot. Exploring initialization methods( xavier initialization, kaiming initialization) and optimization methods(SGD, SGD momentum, Adam) by hand.

4. CNN. Training the model using pytorch lightning framework. Define lightning module(init param, configure_optimizer, training_step, validation_step, test_step). Train using trianer.fit(lightning, train_dataloader). Test using trainer.test(lightning, test_dataloader. Task: CIFAR10 image classification task. Network: GoogleNet(Inception block), ResNet, DenseNet.

5. Transformer. Scaled dot product attention and Multi-Head Attention Layer. The Transformer architecture is based on the Multi-Head Attention layer and applies multiple of them in a ResNet-like block(Layernorm). Explore positional encoding(Since transformer is permutation-equivariant and learning rate warm up(gradient). Usage example: Sequence-to-Sequence problem(reverse the sequence). Set Anomally Detection(no need for positional encoding).

6. RNN. Grpha Convolution Network(GCN) and Graph Attention Network(GAT). Tasks: Node-level task(node classification). Graph-level task(graph classfication).

7. Energy_Generative. Dataset are viewed as samplers of true data distribution which we don't know. We choose to use a Energy-based Generative Model to simulate that probability distribution. Using MCMC(Markov Chain Monte Carlo) sampling to generate fake images from CNN model. MCMC would start at a random vector and update using the gradient of CNN model at this point. Then, we can use contrastive divergence loss (L_DC) between output of real images and output of fake images to train the model and update the parameter of the model. Finally, the model can generate images similar to images in the dataset.

8. AutoEncoder. Use CNN network as encoder to encode the images as latent vectors(latent_dims). Then using transpose convolution network as decoder to reconstruct the images. Using MSE as loss function to measure the distance between input images and reconstructed images. When using small latent_dims, the reconstructed images are blur and reflect rough shape.

9. SimCLR. Contrastive Learning using SimCLR method. The task is to find the representations of images with no labels. We make data augment of the image and try to max the similarity of representations within classes and min it cross classes. So we use InfoNCE loss function to train the Resnet and use the output vector of Resnet as the representations. To prove it's a good representation for downsteam tasks, we use the presenetation to perform Logistic Regression using just 1 Linear layer network. It turns out that the classification accuracy is not bad thus the representation does work.

10. Adversarial. We focus on adversarial attack based on that CNN can be fooled by slight modifications to the input. If you add some designed noise or patches into the images, although the images looks similar to the original image, the class prediction of CNN model would be totally wrong. It can be used to improving a robust model or GAN.
