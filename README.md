## Individual animal identification
---
This study develops a novel three-setep approach integrating segmentation, orientation, and similarity-learning networks to identify individual animals of four species, namely humpback whales, bottlenose dolphins, harbor seals, and western leopard toads.

1. The segmentation network is a [Mask RCNN model](https://github.com/matterport/Mask_RCNN) adapted to our study-case datasets. This model is tasked with isolating and cropping animal objects conveying individual-specific characteristics.
2. The orientation network is a CNN-based model that decides which side of the animal was photographed. This plays a crucial role in individual identification since some key-distinctive features can only appear on one side. The folder [Orientation_models](https://github.com/kabuga1987/Individual_ID/tree/main/Orientation_models) contains the code for training and testing the orientation model as well as a subset of images for two species.

3. The similarity-learning network is either a triplet or a Siamese neural network that takes two images of the same orientation and decides whether they are from the same individual or two different individuals based on similarities. The folder [Similarity_learning_models](https://github.com/kabuga1987/Individual_ID/tree/main/Similarity_learning_models) has all the code for training and testing the similarity-learning models and a subset of images for each species.

This code accompanies the paper: **Similarity learning networks uniquely identify individuals of four marine and terrestrial species**

## Authors
Emmanuel Kabuga, Izzy Langley, Monica Arso Civil, John Measey, Bubacarr Bah, Ian Durbach

## Requirements


All required packages can be installed  using `pip install -r requirements.txt`

Numpy

Pandas

Scipy

Sklearn

PIL

Tensorflow

Keras

## Photo credits

 We thank the individuals and institutions who gave permission to use the images appearing in this repository. Photo credits: Sea Mammal Research Unit, University of St Andrews (all bottlenose dolphin and harbour seal images), Happywhale.org (all humpback whale images), ToadNUTS (all western leopard toad images).









