import os
import sys
import argparse


prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker, ExternalTracker


def run_focal(tracker_name, tracker_param, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video")
    external_tracker = ExternalTracker(tracker_name, tracker_param, "video")
    tracker.run_focal(optional_box=optional_box, debug=debug, save_results=save_results,
                      external_tracker=external_tracker)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_focal(args.tracker_name, args.tracker_param, args.optional_box, args.debug, args.save_results)


if __name__ == '__main__':
    main()
