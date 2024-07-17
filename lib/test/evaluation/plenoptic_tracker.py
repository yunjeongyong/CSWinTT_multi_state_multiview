import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2
import numpy as np
import lmdb


LMDB_ENVS = dict()
LMDB_HANDLES = dict()
LMDB_FILELISTS = dict()


def get_lmdb_handle(name):
    global LMDB_HANDLES, LMDB_FILELISTS
    item = LMDB_HANDLES.get(name, None)
    if item is None:
        env = lmdb.open(name, readonly=True, lock=False, readahead=False, meminit=False)
        LMDB_ENVS[name] = env
        item = env.begin(write=False)
        LMDB_HANDLES[name] = item

    return item

def decode_img(lmdb_fname, key_name):
    handle = get_lmdb_handle(lmdb_fname)
    binfile = handle.get(key_name.encode())
    if binfile is None:
        print("Illegal data detected. %s %s" % (lmdb_fname, key_name))
    s = np.frombuffer(binfile, np.uint8)
    x = cv2.cvtColor(cv2.imdecode(s, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return x


class PlenopticTracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset=None):
        self.name = name
        self.parameter_name = parameter_name
        self.dataset = dataset

        env = env_settings()

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))

        # tracker_module_abspath = "E:\\code\\tracker_assignment\\다중객체추적모델(한성대)\\CSWinTT_multi_state\\lib\\tracker\\cswintt.py"
        print("tracker_module_absapth", tracker_module_abspath)
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.trackers = []

    def replay(self, bboxes, frame_start_idx, frame_end_idx):

        is_first_frame = True
        tracker_num = len(bboxes)
        print("bboxes length:", len(bboxes))
        best_idx = [0 for _ in range(tracker_num)]
        d_index = [5 for _ in range(tracker_num)]  # d: 최적 포컬 인덱스에서 체크할 주변 인덱스 수, k-d ~ k+d
        baseline = False
        last_focal_num = 100

        def _build_init_info(box):
            return {'init_bbox': box}

        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name

        for bbox in bboxes:
            tracker = self.create_tracker(params)
            img, _ = self.dataset.__getitem__(frame_start_idx)
            tracker.initialize(img, _build_init_info(bbox))
            self.trackers.append(tracker)

        return_state = [0, 0, 0, 0]
        for frame_idx in range(frame_start_idx + 1, frame_end_idx + 1):
            frame, focals = self.dataset.__getitem__(frame_idx)
            for k, tracker in enumerate(self.trackers):
                best_state = [0, 0, 0, 0]
                max_score = 0
                if is_first_frame or baseline:
                    range_focals = focals
                elif d_index[k] < best_idx[k] < last_focal_num - d_index[k]:
                    range_focals = focals[best_idx[k] - d_index[k]: best_idx[k] + d_index[k] + 1]
                    # Focal plane range 범위 벗어나지 않게 처리
                elif best_idx[k] <= d_index[k]:
                    range_focals = focals[0: d_index[k] * 2 + 1]
                else:
                    range_focals = focals[-(d_index[k] * 2 + 1):]

                for f_idx, f in enumerate(range_focals):
                    if f_idx == len(range_focals) - 1:
                        tracker.last_sequence = True
                    f_idx = int(f.split('\\')[-1][:3])
                    img = cv2.imread(f)

                    out = tracker.track_multi_state(img, f'{frame_idx}_{f_idx}')
                    state = [int(s) for s in out['target_bbox']]
                    score = float(out['conf_score'])
                    if max_score < score:
                        max_score = score
                        best_idx[k] = int(f.split('\\')[-1][:3])
                        best_state = state
                        return_state = best_state

                if max_score > 0.5:
                    d_index[k] = 3
                else:
                    d_index[k] = 5

        return return_state

    def create_tracker(self, params):
        tracker = self.tracker_class(params, "video")
        return tracker

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv2.imread(image_file)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
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


