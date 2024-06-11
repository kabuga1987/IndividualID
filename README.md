## Individual animal identification
---
This study develops a novel three-setep approach integrating segmentation, orientation, and similarity-learning networks to identify individual animals of four species, namely humpback whales, bottlenose dolphins, harbor seals, and western leopard toads.

1. The segmentation network, [based on Mask RCNN model](https://github.com/matterport/Mask_RCNN), is tailored to isolate and crop animal objects conveying individual-specific characteristics.
2. The orientation network, based on CNNs, determines the photographed side of the animal, which is vital for identifying individuals, as distinctive features may only be present on a specific side. The folder [Orientation_models](https://github.com/kabuga1987/Individual_ID/tree/main/Orientation_models) contains the code for training and testing the orientation model, along with a subset of images for two species.

3. The similarity-learning network, which can be a triplet or Siamese neural network, compares two images of the same orientation to determine if they depict the same individual or different individuals based on visual similarities. The folder [Similarity_learning_models](https://github.com/kabuga1987/Individual_ID/tree/main/Similarity_learning_models) has all the necessary code for training and testing the similarity-learning models and a subset of images for each species.

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

 We thank the individuals and institutions who permitted the public use of the images uploaded to this repository. Photo credits: Sea Mammal Research Unit, University of St Andrews (all bottlenose dolphin and harbour seal images), Happywhale.org (all humpback whale images), ToadNUTS (all western leopard toad images).









