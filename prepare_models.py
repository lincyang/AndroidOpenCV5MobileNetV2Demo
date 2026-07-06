#!/usr/bin/env python3
#-----------------------------------------------------------------------
# <author>程序员Linc</author>
# <wechat>公众号：程序员Linc</wechat>
#-----------------------------------------------------------------------

import argparse
import onnx
import onnx.hub
import pathlib
import shutil
import tempfile
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare MobileNet V2 model files for OpenCV DNN."
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory.",
    )
    return parser.parse_args()


def download_file(url: str, output_path: pathlib.Path):
    with urllib.request.urlopen(url) as response:
        assert response.status == 200
        with open(output_path, mode="wb") as file:
            file.write(response.read())


def download_onnx_model(model_name: str, output_path: pathlib.Path):
    model = onnx.hub.load(model_name)
    onnx.save(model, str(output_path))


def fix_batch_size(model_path: pathlib.Path):
    model = onnx.load(str(model_path))
    for tensor in model.graph.input:
        tensor_type = tensor.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dim in tensor_type.shape.dim:
            if dim.dim_param == "batch_size":
                dim.ClearField("dim_param")
                dim.dim_value = 1
    onnx.save(model, str(model_path))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    download_file(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        args.output_dir / "imagenet_classes.txt",
    )

    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        for model_name, model_path_stem in [
            ("MobileNet v2-1.0-int8", "mobilenetv2_int8"),
            ("MobileNet v2-1.0-fp32", "mobilenetv2_fp32"),
        ]:
            model_path = temp_dir / f"{model_path_stem}.onnx"
            download_onnx_model(model_name, model_path)
            fix_batch_size(model_path)
            dest_model_path = args.output_dir / model_path.name
            shutil.copyfile(model_path, dest_model_path)


if __name__ == "__main__":
    main()
