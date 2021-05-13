import logging

import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
LOGGER = logging.getLogger(__name__)


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTModel:
    def __init__(self, model_path, engine_path, input_shape, plugin_path=None):
        self.model_path = model_path
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.plugin_path = plugin_path

    def build_engine(self, trt_logger, batch_size):
        with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, trt_logger) as parser:

            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = batch_size
            LOGGER.info('Building engine with batch size: %d', batch_size)
            LOGGER.info('This may take a while...')

            if builder.platform_has_fast_fp16:
                LOGGER.info('fp16 mode enabled')
                builder.fp16_mode = True

            # parse model file
            with open(self.model_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    LOGGER.critical('Failed to parse the ONNX file')
                    for err in range(parser.num_errors):
                        LOGGER.error(parser.get_error(err))
                    return None

            network.get_input(0).shape = [batch_size, *self.input_shape]

            if self.plugin_path:
                network = self.add_plugin(network)
            engine = builder.build_cuda_engine(network)
            if engine is None:
                LOGGER.critical('Failed to build engine (maybe caused by batch_size)')
                return None

            LOGGER.info("Completed creating engine")
            with open(self.engine_path, 'wb') as engine_file:
                engine_file.write(engine.serialize())
            return engine

    def add_plugin(self, network):
        raise NotImplementedError()


class TRTInferenceBackend:
    # initialize TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    model: TRTModel

    def __init__(self, model: TRTModel, batch_size=1):
        self.model = model
        self.batch_size = batch_size

        # load plugin if the model requires one
        if self.model.plugin_path is not None:
            try:
                ctypes.cdll.LoadLibrary(self.model.plugin_path)
            except OSError as err:
                raise RuntimeError('Plugin not found') from err

        # load trt engine or build one if not found
        if not self.model.engine_path.exists():
            self.engine = self.model.build_engine(self.TRT_LOGGER, self.batch_size)
        else:
            runtime = trt.Runtime(self.TRT_LOGGER)
            with open(self.model.engine_path, 'rb') as engine_file:
                buf = engine_file.read()
                self.engine = runtime.deserialize_cuda_engine(buf)
        if self.engine is None:
            raise RuntimeError('Unable to load the engine file')
        if self.engine.has_implicit_batch_dimension:
            assert self.batch_size <= self.engine.max_batch_size

        # allocate buffers
        self.bindings = []
        self.outputs = []
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # append the device buffer to device bindings
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                if not self.engine.has_implicit_batch_dimension:
                    assert self.batch_size == shape[0], f'You may be using different batch size than when compiling the engine. {self.batch_size} != {shape[0]}'
                self.input = HostDeviceMem(host_mem, device_mem)
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    @property
    def input_handle(self):
        return self.input.host

    @input_handle.setter
    def input_handle(self, val):
        self.input.host[:] = val

    def infer(self):
        self.infer_async()
        return self.synchronize()

    def run(self, input, *args, **kwargs):
        """This function provides compatible interface."""
        self.input_handle = input.ravel()
        return self.infer()

    def infer_async(self):
        cuda.memcpy_htod_async(self.input.device, self.input.host, self.stream)
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings,
                                       stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

    def synchronize(self):
        self.stream.synchronize()
        return [out.host for out in self.outputs]
