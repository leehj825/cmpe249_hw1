import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from moviepy.editor import ImageSequenceClip

# Specify the path to model config and checkpoint file
config_file = 'fcos_x101-64x4d_fpn_gn-head-1x_waymococo_hj.py'
checkpoint_file = 'work_dirs/fcos_x101-64x4d_fpn_gn-head-1x_waymococo_hj/20230926_224612/best_coco_bbox_mAP_epoch_10.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')

# Test a video and show the results
# Build test pipeline
model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# The interval of show (ms), 0 is block
wait_time = 1

video_reader = mmcv.VideoReader('demo/demo.mp4')

frames = []
#for frame in track_iter_progress(video_reader):
for frame in video_reader:
    result = inference_detector(model, frame, test_pipeline=test_pipeline)
    visualizer.add_datasample(
        name='video',
        image=frame,
        data_sample=result,
        draw_gt=False,
        show=False)
    frame = visualizer.get_image()
    frames += [frame]

clip = ImageSequenceClip(frames, fps=5)
clif_file = 'demo/demo_updated.gif'
clip.write_gif(clif_file, fps=5)
cv2.destroyAllWindows()