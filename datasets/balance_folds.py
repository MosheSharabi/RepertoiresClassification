import os
import argparse

def analyze(args):
    folds_dic = {'f0': [0, 0]}
    for f in os.listdir(args.cubes_path + 'dataINorder/'):
        filename = os.fsdecode(f)
        if('ill' in filename):
            is_ill = 1
        else:
            is_ill = 0

        start = filename.find('_') + 1
        if (filename[start: start + 2] in folds_dic):
            folds_dic[filename[start: start + 2]][is_ill] += 1
        else:
            folds_dic[filename[start: start + 2]] = [0,0]
            folds_dic[filename[start: start + 2]][is_ill] += 1

    print(folds_dic)

def balance(args):
    folds_dic = {'f0': [0, 0, 0, 0, 1]} # {'foldname':[not-ill count, ill count, not-ill max round, ill max round, is not balance]}
    for f in os.listdir(args.cubes_path + 'dataINorder/'):
        filename = os.fsdecode(f)
        if('ill' in filename):
            is_ill = 1
        else:
            is_ill = 0

        # count ILL/no-ill in folds
        start = filename.find('_') + 1
        fold_name = filename[start: start + 2]
        if (fold_name in folds_dic):
            folds_dic[fold_name][is_ill] += 1
        else:
            folds_dic[fold_name] = [0, 0, 0, 0, 1]
            folds_dic[fold_name][is_ill] += 1

        # find max round of ill/not-ill in folds
        round = int(filename[0: start - 1])
        if(round > folds_dic[fold_name][is_ill + 2]):
            folds_dic[fold_name][is_ill + 2] = round

    print(folds_dic) # before

    # start erasing
    while(sum(folds_dic[i][4] for i in folds_dic)):
        for f in os.listdir(args.cubes_path + 'dataINorder/'):
            filename = os.fsdecode(f)
            if('ill' in filename):
                is_ill = 1
            else:
                is_ill = 0

            start = filename.find('_') + 1
            fold_name = filename[start: start + 2]
            round = int(filename[0: start - 1])
            if(folds_dic[fold_name][4]):
                if((folds_dic[fold_name][0] > folds_dic[fold_name][1])):
                    if((is_ill == 0) & (round == folds_dic[fold_name][2])):
                        os.remove(args.cubes_path + "dataINorder/" + filename)
                        os.remove(args.cubes_path + "familiesINorder/" + filename)
                        os.remove(args.cubes_path + "JgenesINorder/" + filename)
                        os.remove(args.cubes_path + "lengthsINorder/" + filename)
                        os.remove(args.cubes_path + "VgenesINorder/" + filename)
                        folds_dic[fold_name][0] -= 1
                elif((folds_dic[fold_name][0] < folds_dic[fold_name][1])):
                    if((is_ill == 1) & (round == folds_dic[fold_name][3])):
                        os.remove(args.cubes_path + "dataINorder/" + filename)
                        os.remove(args.cubes_path + "familiesINorder/" + filename)
                        os.remove(args.cubes_path + "JgenesINorder/" + filename)
                        os.remove(args.cubes_path + "lengthsINorder/" + filename)
                        os.remove(args.cubes_path + "VgenesINorder/" + filename)
                        folds_dic[fold_name][1] -= 1
                else:
                    folds_dic[fold_name][4] = 0
        # update max round
        for i in folds_dic:
            if ((folds_dic[i][0] > folds_dic[i][1])):
                folds_dic[i][2] -= 1
            elif ((folds_dic[i][0] < folds_dic[i][1])):
                folds_dic[i][3] -= 1

    print(folds_dic)    # after


default_cubes_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/Cleaned/cubes/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cubes_path', help='cubes path', type=str, default=default_cubes_path)
    args = parser.parse_args()


    print('start balance folds')
    # analyze(args)
    balance(args)
    print('end')
