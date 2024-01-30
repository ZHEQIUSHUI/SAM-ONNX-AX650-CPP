import onnxruntime
import random
import numpy as np
class onnx_inferencer:

    def __init__(self, model_path) -> None:
        self.onnx_model_sess = onnxruntime.InferenceSession(model_path)
        self.output_names = []
        self.input_names = []
        print(model_path)
        for i in range(len(self.onnx_model_sess.get_inputs())):
            self.input_names.append(self.onnx_model_sess.get_inputs()[i].name)
            print("    input:", i,
                  self.onnx_model_sess.get_inputs()[i].name,self.onnx_model_sess.get_inputs()[i].type,
                  self.onnx_model_sess.get_inputs()[i].shape)

        for i in range(len(self.onnx_model_sess.get_outputs())):
            self.output_names.append(
                self.onnx_model_sess.get_outputs()[i].name)
            print("    output:", i,
                  self.onnx_model_sess.get_outputs()[i].name,self.onnx_model_sess.get_outputs()[i].type,
                  self.onnx_model_sess.get_outputs()[i].shape)
        print("")

    def get_input_count(self):
        return len(self.input_names)

    def get_input_shape(self, idx: int):
        return self.onnx_model_sess.get_inputs()[idx].shape

    def get_input_names(self):
        return self.input_names

    def get_output_count(self):
        return len(self.output_names)

    def get_output_shape(self, idx: int):
        return self.onnx_model_sess.get_outputs()[idx].shape

    def get_output_names(self):
        return self.output_names

    def inference(self, tensor):
        return self.onnx_model_sess.run(
            self.output_names, input_feed={self.input_names[0]: tensor})

    def inference_multi_input(self, tensors: list):
        inputs = dict()
        for idx, tensor in enumerate(tensors):
            inputs[self.input_names[idx]] = tensor
        return self.onnx_model_sess.run(self.output_names, input_feed=inputs)

backbone = onnx_inferencer("../onnx_models/vit_b_dec_simple-point-enc.onnx")

# for i in range(16):
#     tensor = [
#         [random.randint(0,1024),random.randint(0,1024)],
#         [random.randint(0,1024),random.randint(0,1024)],
#         [random.randint(0,1024),random.randint(0,1024)],
#     ]

#     tensor[0][0] = (tensor[1][0]+tensor[2][0])/2
#     tensor[0][1] = (tensor[1][1]+tensor[2][1])/2

#     tensor = np.array(tensor, dtype=np.float32).reshape((1,3,2))
#     tensor_lab = np.array([1,2,3], dtype=np.float32).reshape((1,3))
#     print(tensor, tensor_lab)

#     out = backbone.inference_multi_input([tensor, tensor_lab])
#     print(out[0].shape)
#     np.save("out_rec_%d.npy"%i, out[0])


# for i in range(16):
#     tensor = [
#         [random.randint(0,1024),random.randint(0,1024)],
#         [0,0],
#         [0,0],
#     ]

#     tensor = np.array(tensor, dtype=np.float32).reshape((1,3,2))
#     tensor_lab = np.array([1,0,0], dtype=np.float32).reshape((1,3))
#     print(tensor, tensor_lab)

#     out = backbone.inference_multi_input([tensor, tensor_lab])
#     print(out[0].shape)
#     np.save("out_pt_%d.npy"%i, out[0])