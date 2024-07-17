import importlib
import os
from collections import OrderedDict, deque
from lib.test.evaluation.environment import env_settings
import time
import cv2
import csv
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import time
from copy import deepcopy
from FocalDataloader import LoadFocalFolder
from typing import List


def csv_write(save_path, bbox, save_name):
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f'{save_name}.csv'
    f = open(file_path, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(bbox)
    f.close()

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids=None, display_name: str = None,
                result_only=False):

    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}/{}'.format(env.results_path, "LaSOT", self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(deepcopy(params), self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)
            if frame_num == 80:
                print()

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name

        success, frame = cap.read()

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break
        i = 0
        while True:
            i += 1
            print(f'{i}th frame start!')
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.imwrite(f'result/{i:03d}.png', frame_disp)
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def draw_bboxes(self, img, tracker_num, bbox, color, identities=None, offset=(0, 0), rest_bboxes=None):
        x, y, w, h = bbox
        label = str(tracker_num)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]  # Nonvideo3
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)  # Nonvideo3
        cv2.rectangle(img, (x, y), (x + t_size[0], y + t_size[1]), color, -1)
        cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  # Nonvideo3

        if rest_bboxes is not None:
            for rest_bbox in rest_bboxes:
                tracker_num, bbox = rest_bbox
                self.draw_bboxes(img, tracker_num, bbox, color)

        return img

    def external_init(self, plenoptic_save_path: str, timestamp: str, video_name: str, start_frame_num=0, last_frame_num=100,
                      start_focal_num=0, last_focal_num=100, buffer_size=10):
        self.externals = {}
        self.start_frame_num = start_frame_num
        self.start_focal_num = start_focal_num
        self.last_focal_num = last_focal_num
        self.buffer_size = buffer_size
        self.plenoptic_save_path = os.path.join(plenoptic_save_path, timestamp)
        os.mkdir(self.plenoptic_save_path)

        self.focalDataloader = LoadFocalFolder(video_name, 'focal', frame_range=(start_frame_num, last_frame_num),
                                               focal_range=(start_focal_num, last_focal_num))

    def _external_idx(self, frame_idx, *args):

        return frame_idx - self.start_frame_num + 1 - sum(args)

    def external_init_call(self, multiview_tracker_idx, frame_idx, bbox_buffer, score_buffer: List[deque]):

        def _build_init_info(box):
            return {'init_bbox': box}

        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        tracker = self.create_tracker(params)

        img, _ = self.focalDataloader.__getitem__(self._external_idx(frame_idx, len(bbox_buffer)))
        tracker.initialize(img, _build_init_info(bbox_buffer[0]))

        for i in range(len(bbox_buffer)):
            x, y, w, h = bbox_buffer[i]
            r = i * 25
            color = (r, 0, 255 - r)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 15 if i == 0 else 3)

            label = str(i)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x, y), (x + t_size[0], y + t_size[1]), color, -1)
            cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        end_frame_idx = frame_idx + self.buffer_size + self.buffer_size // 2
        cv2.imwrite(os.path.join(self.plenoptic_save_path,
                                 f"{multiview_tracker_idx}_{frame_idx - len(bbox_buffer)}_init__{end_frame_idx}.png"), img)

        self.externals[multiview_tracker_idx] = {
            "tracker": tracker,
            "is_first_frame": True,
            "start_frame_idx": frame_idx,
            "end_frame_idx": end_frame_idx,
            "best_state": [0, 0, 0, 0],
            "best_idx": 0,
            "prev_max_score": 0,
            "max_score": 0,
            "d_index": 5,
            "tracking_time": 0,
            "done": False,
            "past": {"idx": 0, "prev_tracker": None, "before_prev_tracker": None},
        }

        for i in range(frame_idx - len(bbox_buffer) + 1, frame_idx):
            dataloader_idx = self._external_idx(i)
            img, _ = self.focalDataloader.__getitem__(dataloader_idx)
            self.external_run(multiview_tracker_idx, i)

            if self.externals[multiview_tracker_idx]["is_first_frame"]:
                self.externals[multiview_tracker_idx]["is_first_frame"] = False

            score = self.externals[multiview_tracker_idx]["max_score"]
            with open(os.path.join(self.plenoptic_save_path, "conf.csv"), "a+", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{str(i).zfill(3)}-{multiview_tracker_idx}", score])

            img = deepcopy(img)
            x, y, w, h = self.externals[multiview_tracker_idx]["best_state"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.imwrite(os.path.join(self.plenoptic_save_path,
                                     f"{multiview_tracker_idx}_{i}.png"), img)

    def external_call(self, frame_idx):
        plenoptic_done_items = {}
        keys_to_delete = []

        for multiview_tracker_idx in self.externals.keys():
            self.external_run(multiview_tracker_idx, frame_idx)

            ex = {
                "end_frame_idx": self.externals[multiview_tracker_idx]["end_frame_idx"],
                "best_state": [*self.externals[multiview_tracker_idx]["best_state"]],
                "best_idx": self.externals[multiview_tracker_idx]["best_idx"],
                "prev_max_score": self.externals[multiview_tracker_idx]["prev_max_score"],
                "max_score": self.externals[multiview_tracker_idx]["max_score"],
                "tracking_time": self.externals[multiview_tracker_idx]["tracking_time"],
                "done": self.externals[multiview_tracker_idx]["done"],
            }
            if frame_idx >= ex["end_frame_idx"]:
                if ex["max_score"] > 0.2 or ex["max_score"] >= ex["prev_max_score"] * 2:
                    ex["done"] = True
                    keys_to_delete.append(multiview_tracker_idx)
            plenoptic_done_items[multiview_tracker_idx] = ex

        for key in keys_to_delete:
            self.externals.pop(key, None)

        return plenoptic_done_items if len(plenoptic_done_items) > 0 else None

    def external_run(self, id, frame_idx):

        self.externals[id]["past"]["before_prev_tracker"] = self.externals[id]["past"]["prev_tracker"]
        self.externals[id]["past"]["prev_tracker"] = deepcopy(self.externals[id]["tracker"])
        retry_tracker = deepcopy(self.externals[id]["tracker"])

        start = time.time()

        dataloader_idx = self._external_idx(frame_idx)
        img, focals = self.focalDataloader.__getitem__(dataloader_idx)

        max_score = 0

        d_index = int(self.externals[id]["d_index"])
        best_idx = int(self.externals[id]["best_idx"])

        if not self.externals[id]["is_first_frame"]:
            self.externals[id]["tracker"].first_track = False

        if self.externals[id]["is_first_frame"]:
            range_focals = focals
        elif d_index < best_idx < self.last_focal_num - d_index:
            range_focals = focals[best_idx - d_index: best_idx + d_index + 1]
        elif best_idx <= d_index:
            range_focals = focals[0: d_index * 2 + 1]
        else:
            range_focals = focals[-(d_index * 2 + 1):]

        for i, f in enumerate(range_focals):
            if i == len(range_focals) - 1:
                self.externals[id]["tracker"].last_sequence = True
            i = int(f.split('\\')[-1][:3])

            img = cv2.imread(f)

            out = self.externals[id]["tracker"].track_multi_state(img, f'{frame_idx}_{i}')
            state = [int(s) for s in out['target_bbox']]
            score = float(out['conf_score'])
            print(f'[{id}th] focal {i} score: {score}')

            print("bboxxxxxxxxxx222", out['target_bbox'])

            if max_score < score:
                max_score = score
                self.externals[id]["best_idx"] = int(f.split('\\')[-1][:3])
                self.externals[id]["best_state"] = state

        if max_score < self.externals[id]["max_score"] / 2.0:
            max_score_2 = max_score
            best_idx_2 = 0
            best_state_2 = []
            self.externals[id]["d_index"] = 5
            range_focals = focals
            for i, f in enumerate(range_focals):
                if i == len(range_focals) - 1:
                    retry_tracker.last_sequence = True
                i = int(f.split('\\')[-1][:3])
                img = cv2.imread(f)
                out = retry_tracker.track_multi_state(img, f'{frame_idx}_{i}')
                state = [int(s) for s in out['target_bbox']]
                score = float(out['conf_score'])
                if max_score_2 < score:
                    max_score_2 = score
                    best_idx_2 = int(f.split('\\')[-1][:3])
                    best_state_2 = state
            if max_score < max_score_2:
                max_score = max_score_2
                self.externals[id]["best_idx"] = best_idx_2
                self.externals[id]["best_state"] = best_state_2

        print(f'--[{id}th] Best focal {self.externals[id]["best_state"]} score: {max_score}--')

        if max_score > 0.5:
            self.externals[id]["d_index"] = 3
        else:
            self.externals[id]["d_index"] = 5

        self.externals[id]["prev_max_score"] = self.externals[id]["max_score"]
        self.externals[id]["max_score"] = max_score
        self.externals[id]["tracking_time"] = time.time() - start

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



