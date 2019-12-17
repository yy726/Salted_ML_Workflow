import os
from luigi import Task, Parameter, LocalTarget, build, BoolParameter, ExternalTask, IntParameter
from hashlib import sha256
from .image_processing import image_processing
from .train import trainOne
import multiprocessing


class InputImage(ExternalTask):
    '''external task that checks the existence of image
    and then return them as a local target to supply to
    other luigi tasks

    image_path: path for input image'''
    # Name of the image
    image_path = Parameter()

    def output(self):
        return LocalTarget(self.image_path)


class InputXml(ExternalTask):
    '''external task that checks the existence of training
        and testing xml file
        and then return them as a local target to supply to
        other luigi tasks

        image_path: path for input xml'''
    xml_path = Parameter()

    def output(self):
        return LocalTarget(self.xml_path)


class DetectorTraining(Task):
    '''task that checks the hash of training xml file matches that of
    trained model. If not, re-train the model'''

    train_xml_path = Parameter(default = 'data/face_train/face_train.xml')
    test_xml_path = Parameter(default = 'data/face_test/face_test.xml')
    penalty = IntParameter(default = 5)
    out_path = Parameter()

    def get_detector_salt(self):
        return sha256(self.input()[0].open('rb').read().encode()).hexdigest()[:8]

    def requires(self):
        return [InputXml(xml_path = self.train_xml_path),
                InputXml(xml_path = self.test_xml_path)]

    def output(self):
        salt = self.get_detector_salt()
        required_path = './pretrained_models/' + 'detector_' + salt + '.svm'
        return LocalTarget(required_path)

    def run(self):
        salt = self.get_detector_salt()
        required_path = './pretrained_models/' + 'detector_' + salt + '.svm'
        threads = multiprocessing.cpu_count()
        trainOne(num_threads= threads, penalty = int(self.penalty), train_path=os.path.split(self.train_xml_path)[0],
                 test_path=os.path.split(self.test_xml_path)[0],
                 model_name=required_path)
        os.mkdir(self.out_path)


class ImageProcessing(Task):
    """
    Processes individual images and write them to output.

    in_path: Path for input data folder
    out_path: Path for output data folder
    img_name: Image name
    """

    in_path = Parameter(default = "data/asian")
    out_path = Parameter(default = "data/asianresult_12345678")
    img_name = Parameter(default = "400_0.bmp")

    def requires(self):
        return [InputImage(image_path=os.path.join(self.in_path, self.img_name)),
                self.clone(DetectorTraining, out_path=self.out_path)]

    def output(self):
        return LocalTarget(os.path.join(self.out_path, self.img_name))

    def run(self):
        image_processing(image_path = self.input()[0].path, dstimg_path=self.output().path,
                         weight_file='./pretrained_models/age_gender_train_model.hdf5',
                         shape_predictor=r'./pretrained_models/face_detector_train_model.dat')


class Aggregation(Task):
    """
    Aggregates all images, and writes a salted list of image names to disk.
    If input data changes, the computed salt would be different from the one saved on disk,
    and luigi tasks will be triggered to compute tasks that are missing.

    in_path: Path for input data folder
    out_path: Path for output data folder
    """

    in_path = Parameter(default = "data/asian")
    out_default_path = Parameter(default ="data/asianresult")
    train_xml_path = Parameter(default='data/face_train/face_train.xml')

    def get_input_salt(self):
        return sha256("".join(os.listdir(self.in_path)).encode()).hexdigest()[:8]

    def get_detector_salt(self):
        return sha256(LocalTarget(self.train_xml_path).open('rb').read().encode()).hexdigest()[:8]

    def get_salted_SUCCESS(self):
        return os.path.join(self.out_default_path + "_" + self.get_detector_salt(), "_SUCCESS_" + self.get_input_salt())

    def output(self):
        """This output function returns a local target with a salted directory name,
        e.g. LocalTarget("asianresult-12345678/")"""
        return LocalTarget(self.get_salted_SUCCESS())

    def requires(self):
        return [ImageProcessing(in_path=self.in_path,
                                out_path=self.out_default_path  + "_" + self.get_detector_salt(),
                                img_name=img_name)
                for img_name in os.listdir(self.in_path)]

    def run(self):
        with open(self.get_salted_SUCCESS(), "w") as f:
            pass


