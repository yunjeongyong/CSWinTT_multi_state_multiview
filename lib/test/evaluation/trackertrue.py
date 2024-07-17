import importlib
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2
import csv
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import time
from collections import deque
from kalman_filter import kalman_filter, P_0

"""Run the tracker with the focal sequence."""
from imageDataloader import Load2DFolder
from FocalDataloader import LoadFocalFolder
from config_parser import ConfigParser
from datetime import datetime
import sys
from lib.test.evaluation.plenoptic_tracker import PlenopticTracker


def csv_write(save_path, bbox, save_name):
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f'{save_name}.csv'
    f = open(file_path, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(bbox)
    f.close()

def csv_kalman_write(save_path, x_meas, y_meas, x_esti, y_esti, save_name):
    file_path = save_path / f'{save_name}.csv'
    f = open(file_path, 'a', newline='')
    wr = csv.writer(f)
    for k in x_meas.keys():
        for i in range(len(x_meas[k])):
            wr.writerow([k, x_meas[k][i], y_meas[k][i], x_esti[k][i], y_esti[k][i]])
    f.close()

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.img_buffer = []
        self.delta_threshold = 200

        self.buffer_size = 10
        self.buffers: list[deque] = []
        self.score_buffers: list[deque] = []
        self.increasing = True
        self.x_0_list = {}
        self.x_esti_list = {}
        self.P_list = {}

        self.xpos_meas_save = []
        self.ypos_meas_save = []
        self.xpos_esti_save = []
        self.ypos_esti_save = []

        config = ConfigParser('./config.json')
        video_name = config['video_name']
        start_frame_num = config['start_frame_num']
        last_frame_num = config['last_frame_num']
        start_focal_num = config['start_focal_num']
        last_focal_num = config['last_focal_num']

        plenoptic_video_name = "D:\\newvideo1_001_300"
        self.superglue_dataset = Load2DFolder(video_name, 'focal', frame_range=(start_frame_num, last_frame_num),
                                          focal_range=None)
        self.plenoptic_dataset = LoadFocalFolder(plenoptic_video_name, 'focal', frame_range=(start_frame_num, last_frame_num),
                                         focal_range=(start_focal_num, last_focal_num))
        self.plenoptic_tracker = PlenopticTracker("cswintt", "CSWinTT.pth", self.plenoptic_dataset)

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
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

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
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

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

        cap = cv2.VideoCapture(videofilepath)
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

                x, y, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
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

            cv2.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv2.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv2.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv2.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv2.imwrite(f'result/{i:03d}.png', frame_disp)

        cap.release()
        cv2.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def draw_bboxes(self, img, tracker_num, bbox, color, identities=None, offset=(0, 0)):
        x, y, w, h = bbox
        label = str(tracker_num)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]  # Nonvideo3

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)  # Nonvideo3
        cv2.rectangle(img, (x, y), (x + t_size[0], y + t_size[1]), color, -1)
        cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  # Nonvideo3
        return img

    def _bbox_to_coord(self, bbox):
        x, y, w, h = bbox
        v = np.random.normal(0, 15)
        return x + v, y + v, np.array([x + w // 2 + v, y + h // 2 + v])

    def _save_buffer(self, bbox, score, k: int):
        buffer = self.buffers[k]
        score_buffer = self.score_buffers[k]
        if len(buffer) == 0:
            x, y, _ = self._bbox_to_coord(bbox)
            self.x_0_list[k] = np.array([x, 0, y, 0])
        if len(buffer) >= self.buffer_size:
            buffer.popleft()
            score_buffer.popleft()
        buffer.append(bbox)
        score_buffer.append(score)

    def _distance(self, k: int) -> bool:


        bbox = buffer[-1]
        _, _, z_meas = self._bbox_to_coord(bbox)
        if k not in self.x_esti_list:
            self.x_esti_list[k], self.P_list[k] = self.x_0_list[k], P_0

        self.x_esti_list[k], self.P_list[k] = kalman_filter(z_meas, self.x_esti_list[k], self.P_list[k])

        if len(self.xpos_meas_save) == 0:
            self.xpos_meas_save = [[] for _ in range(num_cam)]
            self.ypos_meas_save = [[] for _ in range(num_cam)]
            self.xpos_esti_save = [[] for _ in range(num_cam)]
            self.ypos_esti_save = [[] for _ in range(num_cam)]

        self.xpos_meas_save[k].append(z_meas[0])
        self.ypos_meas_save[k].append(z_meas[1])
        self.xpos_esti_save[k].append(self.x_esti_list[k][0])
        self.ypos_esti_save[k].append(self.x_esti_list[k][2])

        return False

    def _visualize(self, num_cam):
        for i in range(num_cam):
            fig = plt.figure(figsize=(8, 8))
            plt.gca().invert_yaxis()
            plt.scatter(self.xpos_meas_save[i], self.ypos_meas_save[i], s=300, c="r", marker='*', label='Position: Measurements')
            plt.scatter(self.xpos_esti_save[i], self.ypos_esti_save[i], s=120, c="b", marker='o', label='Position: Estimation (KF)')
            plt.legend(loc='lower right')
            plt.title('Position: Meas. v.s. Esti. (KF), Camera[%d]' % i)
            plt.xlabel('X-pos. [m]')
            plt.ylabel('Y-pos. [m]')
            plt.xlim((-10, 3840))
            plt.ylim((2160, -10))
            fig.canvas.draw()
            plt.show()
            plt.pause(0.05)
            plt.clf()


    def _best_score(self):
        max_score, camera_idx = 0, 0
        for i, score_buffer in enumerate(self.score_buffers):
            score = score_buffer[4]
            if score > max_score:
                max_score = score
                camera_idx = i
        return max_score, camera_idx, self.buffers[camera_idx][4]

    def run_focal(self, optional_box=None, debug=None, visdom_info=None, save_results=False):
        config = ConfigParser('./config.json')
        write_bbox = config['write_bbox']
        write_conf = config['write_conf']
        write_time = config['write_time']
        is_record = config['is_record']
        video_name = config['video_name']
        start_frame_num = config['start_frame_num']
        last_frame_num = config['last_frame_num']
        start_focal_num = config['start_focal_num']
        last_focal_num = config['last_focal_num']
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        save_path = Path(config['save_path']) / timestamp
        save_path.mkdir(parents=True, exist_ok=True)
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        output_boxes = []
        ff_list = []


        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)


        focaldataloader = Load2DFolder(video_name, 'focal', frame_range=(start_frame_num, last_frame_num),
                                          focal_range=None)

        def _build_init_info(box):
            return {'init_bbox': box}

        frame_idx = start_frame_num
        bbox_color = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 0], [0, 255, 255], [128, 128, 128], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128], [0, 128, 128], [0, 0, 128], [255, 0, 255], [138, 43, 226]]

        is_first_frame = True
        print("Please type the number of trackers: ")
        tracker_num = int(sys.stdin.readline())
        tracker = []
        best_idx = [0 for _ in range(tracker_num)]
        d_index = [5 for _ in range(tracker_num)]
        baseline = False

        for k in range(tracker_num):
            tracker.append(self.create_tracker(params))
            self.buffers.append(deque([]))
            self.score_buffers.append(deque([]))

        num_cam = 0
        for frame, focal_image, focals, all_frame in focaldataloader:
            print("focals_len", len(focals))
            if frame_idx > start_frame_num:
                is_first_frame = False

            if is_first_frame:
                num_cam = len(focals)
                for k in range(tracker_num):
                    print('framelen', len(frame))
                    cv2.putText(frame[0], 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1.5, (0, 0, 0), 1)

                    x, y, w, h = cv2.selectROI(video_name, frame[0], fromCenter=False)
                    init_state = [x, y, w, h]
                    tracker[k].initialize(frame[0], _build_init_info(init_state))
                    output_boxes.append(init_state)
                    self.img_buffer = [frame[0] for _ in range(self.buffer_size // 2 + 1)]

            print(f'{frame_idx}th frame start!')
            if write_bbox:
                bbox_list = [frame_idx]
            if write_conf:
                conf_list = [frame_idx]
            if write_time:
                time_list = [frame_idx]

            do_plenoptic = False
            camera_overs = 0
            for k in range(tracker_num):
                if not is_first_frame:
                    tracker[k].first_track = False
                if write_time:
                    tracking_time_start = time.time()
                max_score = 0
                best_state = [0, 0, 0, 0]
                range_focals = all_frame


                for i, f in enumerate(range_focals):
                    print("range_focals:", range_focals)
                    if i == len(range_focals) - 1:
                        tracker[k].last_sequence = True
                    a = f.split("\\")
                    path = a[-1]
                    _, frame_, camera = path.split("_")
                    ff = camera.split(".")[0]
                    print(ff)

                    img = cv2.imread(f)
                    print("f:", f)

                    out = tracker[k].track_multi_state(img, f'{frame_idx}_{ff}')
                    state = [int(s) for s in out[
                        'target_bbox']]
                    score = float(out['conf_score'])
                    print(f'[{k + 1}th] focal {i} score: {score}')
                    self.draw_bboxes(frame[0], int(ff), state, bbox_color[int(ff)])


                    self.update_display = True
                    self.distance_threshold = 20000
                    self._save_buffer(state, score, k)
                    if self._distance(k):
                        camera_overs += 1

                    if max_score < score:
                        print('best_score_camera',ff)
                        ff_list.append(ff)
                        max_score = score
                        output_boxes.append(state)
                        best_idx[k] = ff
                        best_state = state

                b = len(output_boxes) + 1
                a = frame[0]
                print(f'--[{k + 1}th] Best focal {best_idx[k]} score: {max_score}--')

                if max_score > 0.5:
                    d_index[k] = 3
                else:
                    d_index[k] = 5
                if write_bbox:
                    bbox_list.extend(best_state)
                if write_conf:
                    conf_list.append(round(max_score, 2))
                if write_time:
                    tracking_time = time.time() - tracking_time_start
                    time_list.append(round(tracking_time, 2))

            _, camera_idx, best_bbox = self._best_score()
            print("camera_overs", camera_overs)
            if camera_overs >= int(float(len(focals)) / 3.0 * 2.0):
                plenoptic_bbox = self.plenoptic_tracker.replay(
                    [best_bbox], frame_idx - 5, frame_idx)
                tracker[camera_idx].initialize(frame[0], _build_init_info(plenoptic_bbox))


            if is_record:
                cv2.imwrite(f'{save_path}/{frame_idx:03d}.png', frame[0])

            if write_bbox:
                csv_write(save_path, bbox_list, 'bbox')
            if write_conf:
                csv_write(save_path, conf_list, 'conf')
            if write_time:
                csv_write(save_path, time_list, 'time')
            frame_idx += 1

            self.img_buffer.append(frame[0])
            show = self.img_buffer.pop(0)
            cv2.imshow(video_name, show)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

        for img in self.img_buffer:
            cv2.imshow(video_name, img)
        max_num = max(set(ff_list), key=ff_list.count)
        print("%d카메라가 가장 높은 점수이며 자주 정확합니다.\n" % int(max_num))

        self._visualize(num_cam)

        cv2.destroyAllWindows()


    def run_2d(self, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        from imageDataloader import Load2DFolder
        from config_parser import ConfigParser
        import torch
        from datetime import datetime
        import cv2

        config = ConfigParser('./config.json')
        write_bbox = config['write_bbox']
        write_time = config['write_time']
        # write_gt = config['write_gt']
        is_record = config['is_record']
        video_name = config['video_name']
        video_type = config['video_type']
        img2d_ref = config['image2d_ref']
        start_frame_num = config['start_frame_num']
        last_frame_num = config['last_frame_num']
        start_focal_num = config['start_focal_num']
        last_focal_num = config['last_focal_num']
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        save_path = Path(config['save_path']) / timestamp
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)



        output_boxes = []

        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        imagedataloader = Load2DFolder(video_name, frame_range=(start_frame_num, last_frame_num))

        def _build_init_info(box):
            return {'init_bbox': box}


        k = 0
        frame_idx = start_frame_num
        max_score = 0
        best_state = None
        best_idx = [0]
        d_index = [5]
        is_first_frame = True

        for frame in imagedataloader:
            if is_first_frame:
                x, y, w, h = cv2.selectROI(video_name, frame, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                is_first_frame = False
                continue

            print(f'{frame_idx}th frame start!')
            tracking_time_start = time.time()
            # Draw box

            out = tracker.track_one_state(frame)
            state = [int(s) for s in out['target_bbox']]
            score = round(out['conf_score'], 2)
            print(f'score: {score}')

            cv2.rectangle(frame, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)
            cv2.imwrite(f'{save_path}/{frame_idx:03d}.png', frame)
            if write_bbox:
                csv_write(save_path, state, 'bbox')

            if write_time:
                csv_write(save_path, [round(time.time()-tracking_time_start, 2)], 'time')
            frame_idx += 1
            cv2.imshow(video_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()


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
