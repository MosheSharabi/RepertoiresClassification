import os
import argparse

def kfold(args):
    ill_list = []
    healthy_list = []

    for f in os.listdir(args.cleaned_path + '/cubes/'):
        filename = os.fsdecode(f)
        if filename.endswith('.npy'):
            a = os.stat(args.cleaned_path + '/cubes/' + filename)
            size = int((a.st_size - 128) / 8)

            if "ill" in filename:
                ill_list.append((size, filename))
            else:
                healthy_list.append((size, filename))

    ill_list.sort()
    healthy_list.sort()

    l_f_v_j = True # flag to rename the matching lengths, Vfamilies, Vgenes and Jgenes

    for i, t in enumerate(ill_list):
        os.rename(args.cleaned_path + "/cubes/" + t[1], args.cleaned_path + "/cubes/f" + str(i % args.k) + "_" + t[1])
        if(l_f_v_j):
            os.rename(args.cleaned_path + "/families/" + t[1], args.cleaned_path + "/families/" + "/f" + str(i % args.k) + "_" + t[1])
            os.rename(args.cleaned_path + "/Jgenes/" + t[1], args.cleaned_path + "/Jgenes/" + "/f" + str(i % args.k) + "_" + t[1])
            os.rename(args.cleaned_path + "/lengths/" + t[1], args.cleaned_path + "/lengths/" + "/f" + str(i % args.k) + "_" + t[1])
            os.rename(args.cleaned_path + "/Vgenes/" + t[1], args.cleaned_path + "/Vgenes/" + "/f" + str(i % args.k) + "_" + t[1])

    for i, t in enumerate(healthy_list):
        os.rename(args.cleaned_path + "/cubes/" + t[1], args.cleaned_path + "/cubes/f" + str(i % args.k) + "_" + t[1])
        if(l_f_v_j):
            os.rename(args.cleaned_path + "/families/" + t[1], args.cleaned_path + "/families/" + "/f" + str(i % args.k) + "_" + t[1])
            os.rename(args.cleaned_path + "/Jgenes/" + t[1], args.cleaned_path + "/Jgenes/" + "/f" + str(i % args.k) + "_" + t[1])
            os.rename(args.cleaned_path + "/lengths/" + t[1], args.cleaned_path + "/lengths/" + "/f" + str(i % args.k) + "_" + t[1])
            os.rename(args.cleaned_path + "/Vgenes/" + t[1], args.cleaned_path + "/Vgenes/" + "/f" + str(i % args.k) + "_" + t[1])

default_cleaned_path_cmv = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/Cleaned'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--k', help='k fold', type=int, default=5)
    parser.add_argument('--cleaned_path', help='cleaned_path', type=str, default=default_cleaned_path_cmv)
    args = parser.parse_args()

    print('start K_Fold. K = ', args.k)
    kfold(args)
    print('end')
