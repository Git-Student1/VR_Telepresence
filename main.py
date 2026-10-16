import cv2
import os
import argparse

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import parameters as params
from models.pvsnet_model import PVSNet
import helperFunctions as helper
from predict_view import getPositionVector


class Telepresence:
    def findCap(self):
        # Establish a connection to the webcam
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            # print(i, cap.isOpened())
            if cap.isOpened():
                return cap
            cap.release()
        print("Failed to find camera.")
        return None

    def create_parallax_with_cam(self,  args, pose_dims ):
        cap = self.findCap()

        if cap is None:
            raise RuntimeError(
                "Could not find camera, check access of the system to the camera. You need to run this on a Linux subsystem. Run 'ls /dev/video*'. If here is no such directory, there is no camera available to Linux. If you run this project in a Linux subsystem, the subsystem may have no access to the camera.")

        # Discard the beginning, as there will be a lot of error messages will be given into the console in some linux environments during that.
        # Without triggering it early the key_hints would not be visible/findable into the console
        cv2.waitKey(10)
        ret, frame = cap.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imshow("", frame)



        theta = 0
        radius = 0.06





        while cap.isOpened():
            ret, frame = cap.read()
            frame = frame[120:120 + 256, 200:200 + 256, :]
            # create example positions
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            theta+=np.pi/50

            # create parallax
            pos = {
                "x":x,
                "y":y,
                "z":0
            }
            frame = self.create_parallax( frame, args, pos, pose_dims)

            # show parallax to user
            cv2.imshow("", frame)
            # wait for key
            key_val = cv2.waitKey(10) & 0XFF

            # break gracefully
            if key_val == ord('q'):
                break
        # Close the image show frame
        cv2.destroyAllWindows()



    def create_parallax(self, frame, args, pos, pose_dims):
        # Preprocess Input
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_input = transform(
            Image.fromarray(frame_rgb)
        ).unsqueeze(0).to(params.DEVICE)

        pos = getPositionVector(pos["x"], pos["y"], pos["z"], args.pitch, args.yaw, args.roll, pose_dims=pose_dims).unsqueeze(
            0).to(params.DEVICE)

        # Inference
        print(
            f"Status: Generating view for Pose (x={args.x}, y={args.y}, z={args.z}, pitch={args.pitch}, yaw={args.yaw}, roll={args.roll})...")
        with torch.no_grad():
            predicted_img = model(img_input, pos)
            predicted_img = torch.clamp(predicted_img, 0, 1)

        # (1, 3, H, W) -> (H, W, 3)
        predicted_np = predicted_img[0].squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Float [0,1] -> uint8 [0,255]
        predicted_np = (predicted_np * 255).astype(np.uint8)

        # RGB -> BGR
        predicted_frame = cv2.cvtColor(predicted_np, cv2.COLOR_RGB2BGR)

        return predicted_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single novel view using PVSNet")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="output_view.png", help="Output image path")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/checkpoint_coco_pvsnet_lite_256x256.pth",
                        help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, choices=["blender", "coco"], default="coco",
                        help="Dataset type to determine pose dimensions (blender=3, coco=6)")
    parser.add_argument("--x", type=float, default=0.0, help="Translation X")
    parser.add_argument("--y", type=float, default=0.0, help="Translation Y")
    parser.add_argument("--z", type=float, default=0.0, help="Translation Z")
    parser.add_argument("--pitch", type=float, default=0.0, help="Rotation Pitch (only for COCO)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Rotation Yaw (only for COCO)")
    parser.add_argument("--roll", type=float, default=0.0, help="Rotation Roll (only for COCO)")

    args = parser.parse_args()

    # Parse config from checkpoint name or arguments
    base_name = os.path.basename(args.checkpoint).replace(".pth", "")
    is_lite = "lite" in base_name

    if "256x256" in base_name:
        height, width = 256, 256
    elif "512x512" in base_name:
        height, width = 512, 512
    else:
        height, width = 256, 256

    pose_dims = 3 if args.dataset == "blender" else 6

    print(f"Status: Loading Model (Resolution: {height}x{width}, Lite: {is_lite}, Pose Dims: {pose_dims})")

    model = PVSNet(total_image_input=params.params_number_input, pose_dims=pose_dims, height=height, width=width,
                   is_lite=is_lite)
    model = helper.load_Checkpoint(args.checkpoint, model, load_cpu=True)
    model.to(params.DEVICE)
    model.eval()
    Telepresence().create_parallax_with_cam(args, pose_dims)


## start with
# python main.py --image C:\Users\Maxim\Downloads\images.jpg --dataset coco --checkpoint ./checkpoint/checkpoint_coco_pvsnet_lite_256x256.pth