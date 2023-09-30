import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector

# Specify the path to model config and checkpoint file
config_file = 'fcos_x101-64x4d_fpn_gn-head-1x_waymococo_hj.py'
checkpoint_file = 'work_dirs/fcos_x101-64x4d_fpn_gn-head-1x_waymococo_hj/20230926_224612/best_coco_bbox_mAP_epoch_10.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')


# Test a single image and show the results
img_path = '000005.jpg'
result = inference_detector(model, img_path)

# Access the predicted instances, bounding boxes and labels
pred_instances = result.pred_instances
bboxes = pred_instances.bboxes.cpu().numpy()
labels = pred_instances.labels.cpu().numpy()

# Visualize the result on the image
out_img = mmcv.imshow_det_bboxes(
    img_path,
    bboxes,
    labels,
    show=False  # Set to False to prevent immediate display of the image
)

# Save the result image
output_path = 'output.jpg'
mmcv.imwrite(out_img, output_path)