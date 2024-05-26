# Keras-OCR Tensorflow GPU

Train Keras-OCR model using custom dataset with CRNN Backbone https://github.com/kurapan/CRNN 

# Getting Started
## Installation 

```
# To install from master
pip install git+https://github.com/faustomorales/keras-ocr.git#egg=keras-ocr

# create conda 
conda create --name tf python=3.9
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# install tensorflow to Utilize CUDA-GPU
pip install tensorflow==2.10.0 keras=2.10.0


# To install from PyPi
pip install keras-ocr
```

# Datasets
The dataset is not included in this repository, and we cannot release the dataset for IP reason.

# Train
 
on third cell declare the path contain dataset (image file and dataset label). Dataset tree much more look like this :
```

 Dataset_Dir/
└──  train/
    │   ├── rec_get.txt (or other label file)
    │   └── image_files/
    └── test/
        ├── rec_get.txt (or other label file)
        └── image_files/
```

for initial training using custom dataset had to use the kurapan-notop (backbone model only) for including all the character and digit by using ```python string.printable``` 




```python
import random
import string
import math
import itertools
import os
import numpy as np
import imgaug
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection
import os
import string 
import keras_ocr
# Function to read labels file
def _read_born_digital_labels_file(labels_filepath, image_folder):
    """Read a labels file and return (filepath, label) tuples.

    Args:
        labels_filepath: Path to labels file
        image_folder: Path to folder containing images
    """
    if not os.path.exists(labels_filepath):
        raise FileNotFoundError(f"Labels file not found: {labels_filepath}")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    with open(labels_filepath, encoding="utf-8-sig") as f:
        labels_raw = [l.strip().split(",") for l in f.readlines()]
        labels = [
            (
                os.path.join(image_folder, segments[0]),
                None,
                ",".join(segments[1:]).strip()[1:-1],
            )
            for segments in labels_raw
        ]
    return labels

```
```python
#make sure to use GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
```python
recognizer = keras_ocr.recognition.Recognizer(alphabet=string.printable) #for including character and digits
recognizer.compile()
```

```python  
#change with your dataset directory 
train_labels_filepath = "path/to/train/labels/gt.txt" 
train_image_folder = "path/to/train/labels/image_files"
test_labels_filepath = "path/to/test/labels/gt.txt"
test_image_folder = "path/to/test/labels/image_files"

try:
    train_labels = _read_born_digital_labels_file(labels_filepath=train_labels_filepath, image_folder=train_image_folder)
    test_labels = _read_born_digital_labels_file(labels_filepath=test_labels_filepath, image_folder=test_image_folder)
except FileNotFoundError as e:
    print(e)
    # Handle the error appropriately, e.g., by exiting or providing a fallback
    train_labels = []
    test_labels = []

# Ensure the labels were loaded before proceeding
if train_labels:
    train_labels = [(filepath, box, word.lower()) for filepath, box, word in train_labels]
else:
    print("Train labels could not be loaded.")

if test_labels:
    test_labels = [(filepath, box, word.lower()) for filepath, box, word in test_labels]
else:
    print("Test labels could not be loaded.")

# Verify if the labels are loaded correctly
print(f'Number of training labels: {len(train_labels)}')
print(f'Number of test labels: {len(test_labels)}')
```

```python
batch_size=16
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0)),
])

# Use the provided training and test labels
training_labels = train_labels
validation_labels = test_labels


# Create image generators and calculate steps per epoch
(training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
    (
        keras_ocr.datasets.get_recognizer_image_generator(
            labels=labels,
            height=recognizer.model.input_shape[1],
            width=recognizer.model.input_shape[2],
            alphabet=recognizer.alphabet,
            augmenter=augmenter if labels is training_labels else None
        ),
        len(labels) // batch_size
    ) for labels in [training_labels, validation_labels]
]

# Create batch generators for training and validation
training_gen, validation_gen = [
    recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=batch_size
    )
    for image_generator in [training_image_gen, validation_image_gen]
]
# Print the number of training and validation images
print(f"Number of training images: {len(training_labels)}")
print(f"Number of validation images: {len(validation_labels)}")
```

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10000, restore_best_weights=False),
    tf.keras.callbacks.ModelCheckpoint('train_25/05/2024.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('recognizer_borndigital1.csv')
]
recognizer.training_model.fit_generator(
    generator=training_gen,
    steps_per_epoch=training_steps,
    validation_steps=validation_steps,
    validation_data=validation_gen,
    callbacks=callbacks,
    epochs=1000,
)
```

```python
404/404 [==============================] - 173s 396ms/step - loss: 12.0437 - val_loss: 4.5014
Epoch 2/1000
404/404 [==============================] - 185s 460ms/step - loss: 3.2170 - val_loss: 4.1807
Epoch 3/1000
404/404 [==============================] - 211s 523ms/step - loss: 2.4911 - val_loss: 3.6136
Epoch 4/1000
404/404 [==============================] - 234s 581ms/step - loss: 1.9699 - val_loss: 4.4879
Epoch 5/1000
404/404 [==============================] - 246s 609ms/step - loss: 1.8767 - val_loss: 3.0545
Epoch 6/1000
404/404 [==============================] - 241s 596ms/step - loss: 1.5575 - val_loss: 3.6937
Epoch 7/1000
404/404 [==============================] - 236s 584ms/step - loss: 1.3168 - val_loss: 3.5552
Epoch 8/1000
404/404 [==============================] - 234s 579ms/step - loss: 2.3404 - val_loss: 4.3794
Epoch 9/1000
404/404 [==============================] - 235s 583ms/step - loss: 4.3467 - val_loss: 6.7924
Epoch 10/1000
404/404 [==============================] - 233s 577ms/step - loss: 2.4592 - val_loss: 4.1862
Epoch 11/1000
404/404 [==============================] - 241s 596ms/step - loss: 2.0802 - val_loss: 3.7456
Epoch 12/1000
404/404 [==============================] - 240s 595ms/step - loss: 1.3665 - val_loss: 3.4022
Epoch 13/1000
404/404 [==============================] - 253s 628ms/step - loss: 1.0741 - val_loss: 3.6471
Epoch 14/1000
404/404 [==============================] - 238s 590ms/step - loss: 0.9260 - val_loss: 3.4695
Epoch 15/1000
404/404 [==============================] - 232s 574ms/step - loss: 0.9573 - val_loss: 3.7888
Epoch 16/1000
404/404 [==============================] - 239s 593ms/step - loss: 1.6163 - val_loss: 5.3737
Epoch 17/1000
404/404 [==============================] - 246s 609ms/step - loss: 2.4600 - val_loss: 4.2255
Epoch 18/1000
404/404 [==============================] - 241s 596ms/step - loss: 2.2385 - val_loss: 4.0596
Epoch 19/1000
404/404 [==============================] - 245s 606ms/step - loss: 1.1426 - val_loss: 3.7951
Epoch 20/1000
404/404 [==============================] - 237s 587ms/step - loss: 0.8813 - val_loss: 3.7877
Epoch 21/1000
404/404 [==============================] - 246s 610ms/step - loss: 0.7392 - val_loss: 4.1039
Epoch 22/1000
404/404 [==============================] - 240s 592ms/step - loss: 1.0709 - val_loss: 3.6097
Epoch 23/1000
404/404 [==============================] - 236s 585ms/step - loss: 6.8307 - val_loss: 21.4362
Epoch 24/1000
404/404 [==============================] - 248s 614ms/step - loss: 5.3659 - val_loss: 4.9911
Epoch 25/1000
404/404 [==============================] - 250s 618ms/step - loss: 5.4367 - val_loss: 4.8680
Epoch 26/1000
404/404 [==============================] - 248s 614ms/step - loss: 3.6088 - val_loss: 4.5393
Epoch 27/1000
404/404 [==============================] - 252s 624ms/step - loss: 2.5042 - val_loss: 7.7912
Epoch 28/1000
404/404 [==============================] - 247s 613ms/step - loss: 1.9474 - val_loss: 4.6266
Epoch 29/1000
404/404 [==============================] - 250s 618ms/step - loss: 2.6140 - val_loss: 5.9632
Epoch 30/1000
404/404 [==============================] - 245s 606ms/step - loss: 3.5311 - val_loss: 4.1049
Epoch 31/1000
404/404 [==============================] - 252s 625ms/step - loss: 1.2208 - val_loss: 3.7918
Epoch 32/1000
404/404 [==============================] - 264s 654ms/step - loss: 2.4513 - val_loss: 3.8759
Epoch 33/1000
404/404 [==============================] - 257s 637ms/step - loss: 2.0319 - val_loss: 4.1609
Epoch 34/1000
404/404 [==============================] - 255s 631ms/step - loss: 0.9157 - val_loss: 3.3528
Epoch 35/1000
404/404 [==============================] - 246s 609ms/step - loss: 0.7453 - val_loss: 3.8988
Epoch 36/1000
404/404 [==============================] - 241s 597ms/step - loss: 0.7029 - val_loss: 4.0758
Epoch 37/1000
404/404 [==============================] - 249s 617ms/step - loss: 0.7375 - val_loss: 3.5062
Epoch 38/1000
404/404 [==============================] - 244s 605ms/step - loss: 0.5552 - val_loss: 3.8738
Epoch 39/1000
404/404 [==============================] - 239s 591ms/step - loss: 0.5023 - val_loss: 3.7701
Epoch 40/1000
404/404 [==============================] - 243s 601ms/step - loss: 0.4888 - val_loss: 3.8928
Epoch 41/1000
404/404 [==============================] - 245s 606ms/step - loss: 6.1638 - val_loss: 4.1757
Epoch 42/1000
404/404 [==============================] - 260s 644ms/step - loss: 4.6799 - val_loss: 4.0983
Epoch 43/1000
404/404 [==============================] - 240s 596ms/step - loss: 0.5380 - val_loss: 3.8523
Epoch 44/1000
404/404 [==============================] - 237s 586ms/step - loss: 0.5604 - val_loss: 3.9146
Epoch 45/1000
404/404 [==============================] - 230s 570ms/step - loss: 0.7845 - val_loss: 3.8062
Epoch 46/1000
404/404 [==============================] - 238s 589ms/step - loss: 0.4286 - val_loss: 3.8595
Epoch 47/1000
404/404 [==============================] - 248s 614ms/step - loss: 0.4351 - val_loss: 3.8263
Epoch 48/1000
404/404 [==============================] - 255s 631ms/step - loss: 0.3818 - val_loss: 3.9586
Epoch 49/1000
404/404 [==============================] - 249s 616ms/step - loss: 0.9000 - val_loss: 3.8094
Epoch 50/1000
404/404 [==============================] - 241s 596ms/step - loss: 0.3705 - val_loss: 4.2002
Epoch 51/1000
404/404 [==============================] - 241s 596ms/step - loss: 0.3319 - val_loss: 4.1298
Epoch 52/1000
404/404 [==============================] - 255s 632ms/step - loss: 0.3237 - val_loss: 3.9485
Epoch 53/1000
404/404 [==============================] - 237s 588ms/step - loss: 0.4149 - val_loss: 5.4270
Epoch 54/1000
404/404 [==============================] - 234s 580ms/step - loss: 0.4403 - val_loss: 4.1818
Epoch 55/1000
404/404 [==============================] - 229s 566ms/step - loss: 0.2971 - val_loss: 4.2257
Epoch 56/1000
404/404 [==============================] - 229s 567ms/step - loss: 0.2846 - val_loss: 4.3026
Epoch 57/1000
404/404 [==============================] - 236s 584ms/step - loss: 0.3030 - val_loss: 4.0200
Epoch 58/1000
404/404 [==============================] - 233s 575ms/step - loss: 0.2791 - val_loss: 4.2941
Epoch 59/1000
404/404 [==============================] - 232s 575ms/step - loss: 1.7140 - val_loss: 5.0784
Epoch 60/1000
404/404 [==============================] - 237s 587ms/step - loss: 0.6201 - val_loss: 3.8140
Epoch 61/1000
404/404 [==============================] - 234s 579ms/step - loss: 0.2826 - val_loss: 4.3676
Epoch 62/1000
404/404 [==============================] - 236s 585ms/step - loss: 2.6333 - val_loss: 6.5297
Epoch 63/1000
404/404 [==============================] - 246s 610ms/step - loss: 1.7653 - val_loss: 4.3197
Epoch 64/1000
404/404 [==============================] - 241s 597ms/step - loss: 0.2605 - val_loss: 4.3826
Epoch 65/1000
404/404 [==============================] - 236s 584ms/step - loss: 0.2693 - val_loss: 4.4032
Epoch 66/1000
404/404 [==============================] - 233s 577ms/step - loss: 0.2425 - val_loss: 4.2565
Epoch 67/1000
404/404 [==============================] - 228s 564ms/step - loss: 0.2223 - val_loss: 4.7537
Epoch 68/1000
404/404 [==============================] - 226s 559ms/step - loss: 0.2106 - val_loss: 4.1143
Epoch 69/1000
404/404 [==============================] - 227s 561ms/step - loss: 0.2060 - val_loss: 4.5977
Epoch 70/1000
404/404 [==============================] - 224s 555ms/step - loss: 0.2001 - val_loss: 3.9640
Epoch 71/1000
404/404 [==============================] - 225s 556ms/step - loss: 0.2315 - val_loss: 5.0090
Epoch 72/1000
404/404 [==============================] - 223s 551ms/step - loss: 0.1970 - val_loss: 4.6689
Epoch 73/1000
404/404 [==============================] - 222s 550ms/step - loss: 0.2152 - val_loss: 4.5862
Epoch 74/1000
404/404 [==============================] - 224s 554ms/step - loss: 0.1902 - val_loss: 4.7689
Epoch 75/1000
404/404 [==============================] - 225s 558ms/step - loss: 0.1852 - val_loss: 4.4780
Epoch 76/1000
404/404 [==============================] - 225s 556ms/step - loss: 0.2089 - val_loss: 4.5127
Epoch 77/1000
404/404 [==============================] - 235s 583ms/step - loss: 0.1786 - val_loss: 4.8652
Epoch 78/1000
404/404 [==============================] - 233s 575ms/step - loss: 0.1735 - val_loss: 4.9892
Epoch 79/1000
404/404 [==============================] - 228s 564ms/step - loss: 0.1837 - val_loss: 4.5519
Epoch 80/1000
404/404 [==============================] - 225s 557ms/step - loss: 0.1673 - val_loss: 4.3493
Epoch 81/1000
404/404 [==============================] - 227s 562ms/step - loss: 0.1581 - val_loss: 4.6533
Epoch 82/1000
404/404 [==============================] - 225s 558ms/step - loss: 0.1865 - val_loss: 4.4416
Epoch 83/1000
404/404 [==============================] - 234s 580ms/step - loss: 0.1396 - val_loss: 4.7405
```

The model compiled is overfitting due to train and test data, it need more variations to decreasing the validation loss. Training using 10000+ images take about 234ms/step using GTX 1050Ti, CUDA 11.6, CuDNN 8.4 

#Test and Validation
Test using the test data, which 1935 image from our private dataset. The test divided into two categories, overall accuracy and character accuracy.  

```python
...................................................
Predicted: kewarganegaraan, Actual: kewarganegaraan
1/1 [==============================] - 0s 75ms/step
Predicted: wni, Actual: wni
1/1 [==============================] - 0s 67ms/step
Predicted: berlaku hingga, Actual: berlakuhangga
1/1 [==============================] - 0s 52ms/step
Predicted: seumur hidup, Actual: seumur hidup
1/1 [==============================] - 0s 67ms/step
Predicted: nara, Actual: nama
1/1 [==============================] - 0s 93ms/step
Predicted: tempat/tgl lahir, Actual: tempal/tgl lahir
Overall Accuracy: 75.29715762273902%
Character Accuracy: 83.85959899033195%
```

