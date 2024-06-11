import sys

sys.path.append(".")

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import click
import cv2
from assertpy.assertpy import assert_that
from centernet.config import add_centernet_config
from config import settings as conf
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo
from python_file import count_files
from python_video import frames_to_video, video_info
from tqdm import tqdm
from unidet.config import add_detic_config


def setup_cfg(args):
    cfg = get_cfg()

    add_centernet_config(cfg)
    add_detic_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later

    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

    cfg.freeze()

    return cfg


root = Path.cwd()
dataset = conf.active.dataset
detector = conf.active.detector
video_in_dir = root / conf[dataset].path
generate_video = conf.unidet.detect.generate_videos
video_out_dir = root / f"data/{dataset}/{detector}/detect/videos"
json_out_dir = root / f"data/{dataset}/{detector}/detect/json"
video_ext = conf[dataset].ext

detic_dir = root / "UniDet"
detic_config = detic_dir / conf.unidet.detect.config
detic_checkpoint = detic_dir / conf.unidet.detect.checkpoint

assert_that(video_in_dir).is_directory().is_readable()
assert_that(detic_config).is_file().is_readable()
assert_that(detic_checkpoint).is_file().is_readable()

mp.set_start_method("spawn", force=True)

args = argparse.ArgumentParser()
args.config_file = detic_config
args.confidence_threshold = conf.unidet.detect.confidence
args.parallel = conf.unidet.detect.parallel
args.opts = ["MODEL.WEIGHTS", str(detic_checkpoint)]

setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

print("Input:", video_in_dir)
print("Output:", json_out_dir)
print("Generate video:", generate_video)
print("Output video:", video_out_dir)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

cfg = setup_cfg(args)
demo = VisualizationDemo(cfg, parallel=conf.unidet.detect.parallel)
n_videos = count_files(video_in_dir, ext=video_ext)
bar = tqdm(total=n_videos)

for file in video_in_dir.glob(f"**/*{video_ext}"):
    action = file.parent.name
    json_out_path = json_out_dir / action / file.with_suffix(".json").name

    if json_out_path.exists() and json_out_path.stat().st_size:
        bar.update(1)
        continue

    video_in = cv2.VideoCapture(str(file))
    n_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    gen = demo.run_on_video(video_in)
    detection_data = {}
    out_frames = []

    for i, (viz, pred) in enumerate(gen):
        bar.set_description(f"{i}/{n_frames}")

        if generate_video:
            rgb = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
            out_frames.append(rgb)

        detection_data.update(
            {
                i: [
                    (pred_box.tolist(), score.tolist(), pred_class.tolist())
                    for pred_box, score, pred_class in zip(
                        pred.pred_boxes.tensor,
                        pred.scores,
                        pred.pred_classes,
                    )
                ]
            }
        )

    video_in.release()
    json_out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_out_path, "w") as json_file:
        json.dump(detection_data, json_file)

    if generate_video:
        video_out_path = video_out_dir / action / file.with_suffix(".mp4").name

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(out_frames, video_out_path, conf.active.video.writer)

    bar.update(1)

bar.close()
