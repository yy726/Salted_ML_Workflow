<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Salted_ML_Workflow](#salted_ml_workflow)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Salted_ML_Workflow
A luigi workflow based face detection package

### Introduction
This is a machine learning project that is incorporated within the luigi framework.
The program automatically detects changes in the training input and testing input to 
determine the required action using [luigi tasks](https://github.com/spotify/luigi).

The tasks are organized in a simplified implementation of salted graph to ensure the 
most recent data is automatically recognized an used, while also tracking previous outputs
in order to avoid repetitive work. 

The end result is a workflow that could take in training data, trains a model that can 
detect faces in a testing image and then predicts the person's gender and age. 

### Demo:
To get started with the project, set up environment with

`pipenv install --ignore-pipfile`

then `pipenv shell` to enter the virtual environment


Test your images with the trained model:

`python -m FaceDetection inputpath outputpath`

For instance:

`python -m FaceDetection data/asian data/asianresult`

The repo needs pretrained models and sample data in order to run, which are not committed 
as part of the repo. For the most up-to-date model, contact Yuheng Chen at yc1800@nyu.edu.
For general questions, watch the tutorial on [YouTube]() or contact the authors at yiy521@g.harvard.edu
or yc1800@nyu.edu.

### Input and output
This project needs two steps in order to achieve the goal above. First, it needs to extract
the faces from an image. Then, cropping and normalizing the detected face, it is then able
to predict the gender and age.
#### Detector
The detector is used to detect the face region on an image. Training images and a .xml file
 containing labels denoting the face locations. 
 
To train the detector with your own images, you can use the luigi workflow automatically or 
manually via:
- putting your training images under `./data/face_train/images` and .xml label file under 
`./data/face_train`. If you want to put them in another place, modify `trainDetector.py` 
codes accordingly
- to accelerate the training process with multiple threads, modify thread count in 
`trainDetector.py`
- enter `python FaceDetection/trainDetector.py`
- you should get your new .svm model in your `pretrained_models` folder

#### Analyzer
After being able to detect faces, we need to be able to predict the gender and age.
To do this manually, use 

`python FaceDetection/create_db.py --output db_name.mat --db database --img_size int`

to create database using your custom database. Default database is open source database "wiki".

Then, train using the generated database:

`python FaceDetection/train.py --input data/db_name.mat`

To finish training. The resulting model should be in `checkpoints` folder.

### Workflow

#### Testing
In an industrial setting, data is fluid and is constantly changing. New data may be
added and old data such as out-dated files could be deleted. 

All input(testing) files are concatenated and hashed to create a (almost) unique salt.
After each run, a _SUCCESS flag is written in the output folder with the salt. If any 
file is added or removed, it will incur a re-run of the testing tasks. Original images 
will not be touched, additional images will be processed, and removed images will cause 
their output to be removed. 

#### Training
Periodically, new data can be added to train the model to be more accurate or to adapt
 to latest characteristics. The luigi workflow of this project also captures the change
 in training data.
 
Training data is also hashed to create a unique trained model file specific to the set of 
training data. Once re-trained, a new folder will be created and all test will be re-run
using the latest model.

### Authors and division of work:
Yuheng Chen:

- Machine learning algorithms and environment packaging
- Project proposal and planning including luigi skeleton design
- Luigi implementation(training)
- Data provision


Darcy Yao:

- Project proposal and planning
- Luigi implementation(testing)
- Tests
- Presentation recording
