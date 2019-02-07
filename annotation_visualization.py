import numpy as np
from PIL import Image
import torch
from FuseNetClass import FuseNet
from torch.autograd import Variable
from utils.data_utils_class import get_data

gpu_device = 0
torch.cuda.set_device(gpu_device)
print('[INFO] Chosen GPU Device: ' + str(torch.cuda.current_device()))

height, width, _ = (240, 320, 3)  # Update this part (!)
width *= 3
new_im = Image.new('RGB', (width, height))


def paint_and_save(image, rgb_image, idx):
        """Explain what this function does.
        """
        x_offset = 0

        image = Image.fromarray(image, mode="P")
        image.convert("P")
        image.putpalette(palette)

        rgb_image = Image.fromarray(rgb_image)
        new_im.paste(rgb_image, (x_offset, 0))
        x_offset += rgb_image.size[0]
        new_im.paste(image, (x_offset, 0))

        new_im.save('/home/markus/FuseNet_Predictions/prediction_' + str(idx+1) + '.png')
        print('[PROGRESS] %i of %i images saved.' % (idx+1, test_size))


palette = []

with open('visualization.txt', 'r') as f:
        lines = f.read().splitlines()

for line in lines:
        colors = line.split(', ')
        for color in colors:
                palette.append(float(color))

palette = np.uint8(np.multiply(255, palette))

test_size = 795
model_path = './models/nyu/model_2.pth.tar'
predictions = np.uint8()

dset_name = 'NYU'

if dset_name == 'NYU':
        seg_classes = 40
else:
        seg_classes = 37

_, test_data = get_data(dset_name=dset_name, use_train=False, use_test=True, use_class=True)
print("[INFO] %s dataset has been retrieved." % dset_name)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
print("[INFO] Test loader for %s dataset has been created." % dset_name)

model = FuseNet(seg_classes)
print("[INFO] FuseNet model has been created. Produces %i segmentation classes." % seg_classes)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
print("[INFO] Weights from pretrained FuseNet model has been loaded from checkpoint: %s" % model_path)

model.eval()

seg_pred_images = []

for i, batch in enumerate(test_loader):
        test_rgb_inputs = Variable(batch[0].cuda(gpu_device))
        test_depth_inputs = Variable(batch[1].cuda(gpu_device))
        test_seg_labels = Variable(batch[2].cuda(gpu_device))
        test_class_labels = Variable(batch[3].cuda(gpu_device))

        test_seg_outputs, _ = model(test_rgb_inputs, test_depth_inputs)

        # test_outputs_class = test_outputs_class.data.cpu().numpy()[0]
        _, test_seg_preds = torch.max(test_seg_outputs, 1)
        # _, test_preds_class = torch.max(test_outputs_class, 1)
        # test_preds_class = torch.max(test_preds_class)
        # test_preds_class = test_preds_class.view(-1)

        test_seg_labels = test_seg_labels - 1
        # test_preds_class += 1
        test_seg_preds = test_seg_preds.data.cpu().numpy()[0]
        test_seg_labels = test_seg_labels.data.cpu().numpy()[0]
        comparison_images = np.hstack((np.uint8(test_seg_labels + 1), np.uint8(test_seg_preds + 1)))

        test_rgb_inputs = test_rgb_inputs.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, ::-1]

        paint_and_save(comparison_images, np.uint8(test_rgb_inputs), i)

print('[COMPLETED] Boring prediction images are now nice and colorful!')
