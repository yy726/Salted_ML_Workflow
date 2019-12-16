import os
from luigi import Task, Parameter, LocalTarget, build, BoolParameter, ExternalTask
from hashlib import sha256
from .image_processing import image_processing


class InputImage(ExternalTask):

    # Name of the image
    image_path = Parameter()

    def output(self):
        return LocalTarget(self.image_path)


class ImageProcessing(Task):
    """
    Processes individual images and write them to output.

    in_path: Path for input data folder
    out_path: Path for output data folder
    img_name: Image name
    """

    in_path = Parameter(default = "data/asian")
    out_path = Parameter(default = "data/asianresult")
    img_name = Parameter(default = "400_0.bmp")

    def requires(self):
        return InputImage(image_path=os.path.join(self.in_path, self.img_name))

    def output(self):
        return LocalTarget(os.path.join(self.out_path, self.img_name))

    def run(self):
        image_processing(image_path = self.input().path, dstimg_path=self.output().path,
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
    out_path = Parameter(default = "data/asianresult")

    def get_salt(self):
        return sha256("".join(os.listdir(self.in_path)).encode()).hexdigest()[:8]

    def get_salted_SUCCESS(self):
        return os.path.join(self.out_path, "_SUCCESS_" + self.get_salt())

    def output(self):
        """This output function returns a local target with a salted directory name,
        e.g. LocalTarget("asianresult-12345678/")"""
        return LocalTarget(self.get_salted_SUCCESS())

    def requires(self):
        return [ImageProcessing(in_path=self.in_path,
                                out_path=self.out_path,
                                img_name=img_name)
                for img_name in os.listdir(self.in_path)]

    def run(self):
        with open(self.get_salted_SUCCESS(), "w") as f:
            pass