import torch


def pytorch_to_onnx(inference_model: torch.nn.Module, 
                    model_path: str, 
                    output_path: str,
                    input_shape,
                    simplify: bool = True):
    """
    inference_model: pytorch model
    model_path: *.pth file with weights
    output_path: exported model filename
    input_shape: model input shape, for example (1, 3, 64, 64), used for dummy input
    simplify: simplify the model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded_model = torch.load(model_path)
    res = inference_model.load_state_dict(loaded_model['state_dict'])
    # should yield "<All keys matched successfully>"
    print('load_state_dict:', res)
    inference_model = inference_model.to(device)
    inference_model.eval()

    dummy_input = torch.randn(input_shape, device='cuda')

    torch.onnx.export(inference_model, 
                  dummy_input, 
                  output_path,
                  input_names=['input_1'],
                  output_names=['output'],
                  verbose=True, 
                  opset_version=11, 
                  dynamic_axes={'input_1': [0], 'output': [0]}
                )

    if simplify:
        import onnx
        from onnxsim import simplify
        model = onnx.load(output_path)
        model_simp, check = simplify(model, input_shapes={'input_1': input_shape}, dynamic_input_shape=True)
        assert check
        onnx.save(model_simp, output_path)
