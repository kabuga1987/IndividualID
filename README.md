## Code implementations accompanying the paper -- **Unique animal identification from images with neural networks**.
---
The paper develops a novel three-setep approach integrating segmentation, orientation, and similarity-learning networks, and demonstrates the use of the approach on four species namely humpack whales, bottlenose dolphins, harbour seals, and western leopard toads.

1. The segmentation network is a [Mask RCNN model](https://github.com/matterport/Mask_RCNN) adapted to our study-case datasets. This model is tasked with isolating and cropping animal objects conveying individual-specific characteristics.
2. The orientation network is a CNN-based model that decides which side of the animal was photographed. This plays a crucial role in individual identification since some key-distinctive features can only appear on one side. The folder [Orientation_models](https://github.com/kabuga1987/Individual_ID/tree/main/Orientation_models) contains the code for training and testing the orientation model as well as a subset of images for two species.

3. The similarity-learning network is either a triplet or a Siamese neural network that takes two images of the same orientation and decides whether they are from the same individual or two different individuals based on similarities. The folder [Similarity_learning_models](https://github.com/kabuga1987/Individual_ID/tree/main/Similarity_learning_models) has all the code for training and testing the similarity-learning models and a subset of images for each species.

All the required packages for running the models can be installed from the `requirements.txt` file.