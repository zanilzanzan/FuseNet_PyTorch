import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
from fusenet_model import FuseNet
from torch.autograd import Variable
from utils.data_utils import get_data
import sys
import os

def paint_and_save(image, rgb_image, scene_label, scene_pred, idx):
        """Function takes a comparision image of semantic segmentation labels, an RGB image, ground-truth and
        predicted scene classification labels, and image index. Produces a comparison image and saves it to the
        corresponding location.
        """
        x_offset = 0

        image = Image.fromarray(image, mode="P")
        image.convert("P")
        image.putpalette(palette)

        rgb_image = Image.fromarray(rgb_image)
        new_image.paste(rgb_image, (x_offset, 0))
        x_offset += rgb_image.size[0]
        new_image.paste(image, (x_offset, 0))

        # Print scene class names on ground truth and prediction
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.load_default().font
        draw.text((330, 10), ('scene class: ' + scene_class_dict[scene_label]), (255, 255, 255), font=font)
        draw.text((650, 10), ('scene class: ' + scene_class_dict[scene_pred]), (255, 255, 255), font=font)

        new_image.save('./prediction_visualization/prediction_' + str(idx+1) + '.png')
        print('[PROGRESS] %i of %i images saved     ' % (idx+1, test_size), end='\r')


if __name__ == '__main__':
    save_path = './prediction_visualization/'  # Take dynamically
    if os.path.exists(save_path):
        key = input('[INFO] Taget directory already exists. You might lose previously saved images. Continue:Abort (y:n): ')
        if not key.lower() == 'y':
            print('[ABORT] Script stopped running. Images have not been saved.')
            sys.exit()
    else:
        os.makedirs(save_path)

    # Load the GPU device
    gpu_device = 0
    torch.cuda.set_device(gpu_device)
    print('[INFO] Chosen GPU Device: ' + str(torch.cuda.current_device()))

    # Create the scene classification ID:NAME dictionary
    scene_class_dict = {1: 'bedroom', 2: 'kitchen', 3: 'living room', 4: 'bathroom',
                        5: 'dining room', 6: 'office', 7: 'home office', 8: 'classroom',
                        9: 'bookstore', 10: 'others'}

    # Read the palette values that will be used for coloring the semantic segmentation labels, from the .txt file
    with open('visualization_palette.txt', 'r') as f:
            lines = f.read().splitlines()
            palette = []

    for line in lines:
            colors = line.split(', ')
            for color in colors:
                    palette.append(float(color))

    palette = np.uint8(np.multiply(255, palette))

    # Read the dataset and create dataset loader
    dset_name = 'NYU'  # Take dynamically
    if dset_name == 'NYU':
            seg_classes = 40
    elif dset_name == 'SUN':
            seg_classes = 37
    else:
        raise Exception('Dataset name should be either NYU or SUN.')

    _, test_data = get_data(dset_name=dset_name, use_train=False, use_test=True, use_class=True)
    print("[INFO] %s dataset has been retrieved." % dset_name)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    print("[INFO] Test loader for %s dataset has been created." % dset_name)

    test_size = test_data.__len__()

    # Read the FuseNet model path that will be used for prediction and load the weights to the initialized model
    model_path = './checkpoints/nyu/model.pth.tar'  # Take this dynamically
    model = FuseNet(seg_classes)
    print("[INFO] FuseNet model (%i segmentation classes) has been initialized." % seg_classes)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("[INFO] Weights from pretrained FuseNet model has been loaded. Checkpoint: %s" % model_path)

    model.eval()
    new_image = Image.new('RGB', (960, 240))

    print("[INFO] Prediction starts. Resulting comparision images will be saved under: %s" % model_path)
    for i, batch in enumerate(test_loader):
            test_rgb_inputs = Variable(batch[0].cuda(gpu_device))
            test_depth_inputs = Variable(batch[1].cuda(gpu_device))
            test_seg_labels = Variable(batch[2].cuda(gpu_device))
            test_class_labels = Variable(batch[3].cuda(gpu_device))

            # Predict the pixel-wise classification and scene classification results
            test_seg_outputs, test_class_outputs = model(test_rgb_inputs, test_depth_inputs)

            # Take the maximum values from the feature maps produced by the output layers, for both segmentation and classification
            # Move the tensors to CPU as numpy arrays
            _, test_seg_preds = torch.max(test_seg_outputs, 1)
            test_seg_preds = test_seg_preds.data.cpu().numpy()[0]
            test_seg_labels = test_seg_labels.data.cpu().numpy()[0]

            _, test_class_preds = torch.max(test_class_outputs, 1)
            test_class_labels = test_class_labels.data.cpu().numpy()[0]
            test_class_preds = test_class_preds.data.cpu().numpy()[0]

            # Horizontally stack the predicted and ground-truth semantic segmentation labels
            comparison_images = np.hstack((np.uint8(test_seg_labels), np.uint8(test_seg_preds + 1)))

            # Move the RGB image from GPU to CPU as numpy array and arrange dimensions appropriately
            test_rgb_inputs = test_rgb_inputs.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, ::-1]

            # Color semantic segmentation labels, print scene classification labels, and save comparison images
            paint_and_save(comparison_images, np.uint8(test_rgb_inputs), test_class_labels, (test_class_preds + 1), i)

    print('[INFO] All %i images have been saved.' % test_size)
    print('[COMPLETED] Boring prediction images are now nice and colorful!')
