import argparse
import os, glob
import pickle
from tqdm import tqdm

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from handobjectdatasets.queries import TransQueries, BaseQueries

from mano_train.exputils import argutils
from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.preprocess import prepare_input, preprocess_frame


def forward_pass_3d(model, input_image, pred_obj=True):
    sample = {}
    sample[TransQueries.images] = input_image
    sample[BaseQueries.sides] = ["left"]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
    sample["root"] = "wrist"
    if pred_obj:
        sample[TransQueries.objpoints3d] = input_image.new_ones(
            (1, 600, 3)
        ).float()
    _, results, _ = model.forward(sample, no_loss=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint",
        default="release_models/obman/checkpoint.pth.tar",
    )
    parser.add_argument(
        "--image_path",
        help="Path to image",
        default="readme_assets/images/can.jpg",
    )
    parser.add_argument(
        "--no_beta", action="store_true", help="Force shape to average"
    )
    args = parser.parse_args()
    argutils.print_args(args)

    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)

    # Initialize network
    model = reload_model(args.resume, opts, no_beta=args.no_beta)

    model.eval()

    print(
        "Input image is processed flipped and unflipped "
        "(as left and right hand), both outputs are displayed"
    )

    # load faces of hand
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]
    
    #======== test ========#
    output_folder = "./output_image/frame_HD_centered_hand_50_left"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_list = glob.glob("/x/dandans/Dataset/frame_HD_centered_hand_50_left/*.jpg")
    image_list.sort()
    for image_path in tqdm(image_list):
        image_name = image_path.split("/")[-1]
        output_name = image_name[:-4] + "_obman"

        fig = plt.figure(figsize=(10, 4))
        fig.clf()
        frame = cv2.imread(image_path)
        frame = preprocess_frame(frame)
        input_image = prepare_input(frame)
        #cv2.imshow("input", frame)
        #cv2.imwrite("./output_image/image_demo_input.jpg", frame)
        img = Image.fromarray(frame.copy())
        hand_crop = cv2.resize(np.array(img), (256, 256))

        noflip_hand_image = prepare_input(hand_crop, flip_left_right=False)
        flip_hand_image = prepare_input(hand_crop, flip_left_right=True)
        noflip_output = forward_pass_3d(model, noflip_hand_image)
        flip_output = forward_pass_3d(model, flip_hand_image)

        flip_verts = flip_output["verts"].cpu().detach().numpy()[0]
        noflip_verts = noflip_output["verts"].cpu().detach().numpy()[0]
        #print(flip_verts)
        ax = fig.add_subplot(1, 2, 1)
        ax.title.set_text("input image")
        image_show = np.flip(frame, axis=2).copy()
        image_show = np.flip(image_show, axis=1).copy()
        plt.imshow(image_show)

        # ax = fig.add_subplot(1, 3, 2, projection="3d")
        # ax.title.set_text("flipped input")
        # displaymano.add_mesh(ax, flip_verts, faces, flip_x=True)
        # if "objpoints3d" in flip_output:
        #     objverts = flip_output["objpoints3d"].cpu().detach().numpy()[0]
        #     displaymano.add_mesh(
        #         ax, objverts, flip_output["objfaces"], flip_x=True, c="r"
        #     )
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.title.set_text("unflipped input")
        displaymano.add_mesh(ax, noflip_verts, faces, flip_x=True)
        if "objpoints3d" in noflip_output:
            objverts = noflip_output["objpoints3d"].cpu().detach().numpy()[0]
            displaymano.add_mesh(
                ax, objverts, noflip_output["objfaces"], flip_x=True, c="r"
            )
        plt.savefig(f"./{output_folder}/{output_name}.jpg")

