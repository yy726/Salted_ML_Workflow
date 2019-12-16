import glob
import os
from shutil import copyfile
from unittest import TestCase

from luigi import Task, build, LocalTarget

from FaceDetection.task import InputImage, Aggregation


class InputImageTests(TestCase):
    def test_output(self):
        self.assertIsInstance(InputImage().output(), LocalTarget)


class SaltedIntegrationTests(TestCase):
    def test_salted_tasks(self):
        """This test tests three test cases."""
        try:
            """Test Case 1: When initial input gets added, task runs and writes result to output"""
            os.mkdir("test_output")
            ag = Aggregation(in_path="test_sample", out_path="test_output")
            build([
                ag
            ], local_scheduler=True)
            assert os.path.exists("test_output/400_0.bmp")

            """Test Case 2: When no new input gets added, salt does not change"""
            success_salt = glob.glob("_SUCCESS*")[0]
            ag = Aggregation(in_path="test_sample", out_path="test_output")
            build([
                ag
            ], local_scheduler=True)
            assert os.path.exists(success_salt)

            """Test Case 3: When new input gets added, salt changes"""
            copyfile("test_backup/00000.png", "test_sample/00000.png")
            ag = Aggregation(in_path="test_sample", out_path="test_output")
            build([
                ag
            ], local_scheduler=True)
            assert os.path.exists("test_output/00000.png")
            assert len(glob.glob("_SUCCESS*")) == 2

        finally:
            os.remove("test_output")
            os.remove("test_sample/00000.png")