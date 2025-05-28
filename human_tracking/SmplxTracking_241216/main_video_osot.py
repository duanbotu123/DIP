from portrait_magic.app import process_portrait_video_osot
import argparse
import torch


if __name__ == '__main__':

    # for stable computation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='./data/test.mp4')
    parser.add_argument('--output_dir', type=str, default='./data/test_dir')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--run_mode', type=int, default=0)
    parser.add_argument('--tracking_mode', type=str, default='body')
    # parser.add_argument('--driving_track_path', type=str, default='aa')
    parser.add_argument('--input_type', type=str, default='video')
    parser.add_argument('--sub_vis', type=str, nargs='+', default=[],
        help='the sub folder lists for visualization')
    args = parser.parse_args()

    process_portrait_video_osot(args.video_path, args.output_dir, with_debug=args.debug, run_mode=args.run_mode, tracking_mode=args.tracking_mode, input_type=args.input_type, sub_vis=args.sub_vis, render=args.render)
