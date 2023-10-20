import os
import onnxruntime as ort
import cv2
import numpy as np
import argparse

from pathlib import Path

def enhance_image(onnx_model_path, input_image_path, maxrange, highpercent, lowpercent, gamma):
    onnx_model_path = onnx_model_path
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    maxrange = maxrange / 10.
    hsvgamma = gamma / 10.

    img_A = cv2.imread(input_image_path)

    img_A = img_A[np.newaxis, :]

    out_pred = session.run(None, {'input_4': img_A.astype(np.float32)})

    fake_B = out_pred[0][0]
    fake_B_o = fake_B

    gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 2] * 0.114
    percent_max = sum(sum(gray_fake_B >= maxrange)) / sum(sum(gray_fake_B <= 1.0))

    max_value = np.percentile(gray_fake_B[:], highpercent)
    if percent_max < (100 - highpercent) / 100.:
        scale = maxrange / max_value
        fake_B = fake_B * scale
        fake_B = np.minimum(fake_B, 1.0)

    gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 2] * 0.114
    sub_value = np.percentile(gray_fake_B[:], lowpercent)
    fake_B = (fake_B - sub_value) * (1. / (1 - sub_value))

    imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(imgHSV)
    S = np.power(S, hsvgamma)
    imgHSV = cv2.merge([H, S, V])
    fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
    fake_B = np.minimum(fake_B, 1.0)

    outputs = fake_B

    filename = os.path.basename(input_image_path)
    img_name = 'enhanced_mbllen.jpg'
    outputs = np.minimum(outputs, 1.0)
    outputs = np.maximum(outputs, 0.0)
    cv2.imwrite(img_name, outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Enhancement')
    parser.add_argument('input_image', type=str, help='Path to the input image')
    parser.add_argument('--maxrange', type=float, default=8, help='Max Range (default: 8)')
    parser.add_argument('--highpercent', type=int, default=96, help='High Percent (default: 96)')
    parser.add_argument('--lowpercent', type=int, default=5, help='Low Percent (default: 5)')
    parser.add_argument('--gamma', type=int, default=8, help='Gamma (default: 8)')

    args = parser.parse_args()

    onnx_model_path = f'{str(Path(__file__).resolve().parents[1].joinpath("weights"))}/MBLLEN/onnx/LOL_img_lowlight.onnx'

    enhance_image(onnx_model_path, args.input_image, args.maxrange, args.highpercent, args.lowpercent, args.gamma)
