from pathlib import Path

import numpy as np
import onnxruntime as rt
import onnxruntime.backend as backend
from onnxruntime.capi import _pybind_state as C


def get_available_providers():
    return C.get_available_providers()


class Model:
    def __init__(self, model_path: Path, so=None):
        assert model_path.exists(), f'{model_path} does not exist.'
        self.sess = rt.InferenceSession(str(model_path))

        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.input_type = self.sess.get_inputs()[0].type

        self.output_name = self.sess.get_outputs()[0].name
        self.output_shape = self.sess.get_outputs()[0].shape
        self.output_type = self.sess.get_outputs()[0].type

    def get_sample_input(self):
        return np.random.random(self.input_shape).astype(np.float32)

    def use_cpu(self):
        self.sess.set_providers(['CPUExecutionProvider'])

    def use_gpu(self):
        self.sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def run(self, input):
        return self.sess.run([self.output_name], {self.input_name: input})
