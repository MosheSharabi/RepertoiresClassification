import os
import argparse
import datetime

default_cmv_tsv_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/cmv2/emerson-2017-natgen/'
default_simulated_tsv_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/simulated_data1/'
default_cleaned_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/Cleaned/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cmv_path', help='cmv_path', type=str, default=default_cmv_tsv_path)
    parser.add_argument('--tsv_path', help='tsv_path', type=str, default=default_cmv_tsv_path)
    parser.add_argument('--cleaned_path', help='cleaned_path', type=str, default=default_cleaned_path)
    parser.add_argument('--end_round', type=int, default=1, help='end_round for arrange_from_cube')
    parser.add_argument('--sleep_time', type=float, default=1)
    parser.add_argument('--GIB_per_obj', type=int, default=1)
    parser.add_argument('--ray_cpus', type=int, default=48)
    args = parser.parse_args()

    a1 = datetime.datetime.now()
    # start simulate data
    # os.system('python3 simulate_data.py --cmv_path ' + args.cmv_path + ' --tsv_path ' + args.tsv_path)
    # start simulate data
    # os.system('python3 PrepareData.py --tsv_path ' + args.tsv_path + ' --dest_path ' + args.cleaned_path)
    # start k folds
    # os.system('python3 K_Fold.py --cleaned_path ' + args.cleaned_path)
    # start arrange by lengths
    os.system('python3 arrange_1D.py --cleaned_path ' + args.cleaned_path + ' --end_round ' + str(args.end_round) +
              ' --sleep_time ' + str(args.sleep_time) + ' --GIB_per_obj ' + str(args.GIB_per_obj) + ' --ray_cpus ' + str(args.ray_cpus))
    # os.system('python3 arrange_by_length.py --cleaned_path ' + args.cleaned_path + ' --end_round ' + str(args.end_round))
    # start balance folds
    os.system('python3 balance_folds.py --cubes_path ' + args.cleaned_path + 'cubes/')

    a2 = datetime.datetime.now()
    print((a2 - a1).total_seconds())