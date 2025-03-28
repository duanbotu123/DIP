import argparse
import os
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_list', type=str, default='./data/vid_list.txt', help='text file that contains all videos paths')
    parser.add_argument('--device_id', type=int, default=1, help='text file that contains all videos paths')
    args = parser.parse_args()

    # lists_file = args.run_list
    # data_dir = 'xjdata'

    # with open(lists_file, 'r') as f:
    #     run_lists = f.read().splitlines()

    # run_lists = ['1414_1','1149_1','1149_2','1149_3','1249_2','1249_3','1249_4','BF1FR30','BF2FR30','WM1FR30','WM2FR30','WM3FR30','YM1FR30']
    # run_lists = ['1276_1']
    run_lists = ['1276_2','1276_3','1276_4']

    for run_list in run_lists:
        # run_args = run_list.split(' ')
        # out_dir = os.path.join('data',  'retrack_' + os.path.basename(run_args[0])[:-4])
        # run_command = 'bash refine_track.sh ' + os.path.join(data_dir, run_args[0]) + ' ' + os.path.join(data_dir, run_args[1]) + ' ' + os.path.join(data_dir, run_args[2]) + ' ' + os.path.join(data_dir, run_args[3]) + ' ' + out_dir + ' ' + str(args.device_id)
        run_command = 'bash tracking.sh '+run_list
        print(run_command)
        subprocess.run(run_command, shell=True)