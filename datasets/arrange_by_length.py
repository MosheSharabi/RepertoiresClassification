import argparse
import numpy as np
import multiprocessing as mp
from l_f_j_v_lists import all_j_f_v_l_percent
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
# ray.init(memory= 60000000000, object_store_memory=60000000000)
ray.init()

# class cubeGenerator_plus_loader_arrange_from_cubes(object):
#     def __init__(self, cleaned_path):
#         self.cleaned_path = cleaned_path
#         self.files_names = self.files_names_func(self.cleaned_path + '/cubes/')
#         self.files_names_in_dataINorder = self.files_names_func(self.cleaned_path + '/cubes/dataINorder')
#
#     def files_names_func(self, path):
#         names_list = []
#         for file in os.listdir(path):
#             if ("united" in file):
#                 continue
#             if (file.endswith('.npy')):
#                 names_list.append(file)
#         return names_list
#
#     def get_next_samples(self, i):
#         for file in self.files_names:
#             # if("HIP" in file): to skip HIP / Keck
#             #     continue
#             # if(os.path.isfile(dataInOrderPath + '/' + str(i) + '_' + file_name + '.npy')):
#             if(str(i) + '_' + file in self.files_names_in_dataINorder):
#                 print(str(i) + '_' + file + '  pass')
#                 continue    # no need to generate same file again
#
#             if (cfg.dataset == 'cmv'):
#                 if ("ill" in file):
#                     label = 1
#                 else:
#                     if(cfg.balance_data):
#                         continue
#                     label = 0
#                 data = np.load(self.cleaned_path + '/cubes/' + file)
#
#                 all_lengths = np.load(self.cleaned_path + '/lengths/' + file)
#                 all_Vgenes = np.load(self.cleaned_path + '/Vgenes/' + file)
#                 all_Jgenes = np.load(self.cleaned_path + '/Jgenes/' + file)
#                 all_families = np.load(self.cleaned_path + '/families/' + file)
#                 randnums = np.random.randint(0, len(data), cfg.seq_in_samples)
#                 yield data[randnums], label, file[0:-4], self.cleaned_path + '/cubes/', all_lengths[randnums], all_Vgenes[randnums], all_Jgenes[randnums], all_families[randnums]
#
#             if (cfg.dataset == 'biomed'):
#                 data = np.load(self.cleaned_path + '/cubes/' + file)
#                 # will couse bug, label need be one hot vectore
#                 if("SC" in file):
#                     if(cfg.balance_data):
#                         continue
#                     label = 2
#                 elif("HC" in file):
#                     label = 1
#                 elif("H" in file):
#                     label = 0
#                 else:
#                     print("EROR in generate_all_cubes in utils")
#
#                 all_lengths = np.load(self.cleaned_path + '/lengths/' + file)
#                 all_Vgenes = np.load(self.cleaned_path + '/Vgenes/' + file)
#                 all_Jgenes = np.load(self.cleaned_path + '/Jgenes/' + file)
#                 all_families = np.load(self.cleaned_path + '/families/' + file)
#                 randnums = np.random.randint(0, len(data), cfg.seq_in_samples)
#                 yield data[randnums], label, file[0:-4], self.cleaned_path + '/cubes/', all_lengths[randnums], all_Vgenes[randnums], all_Jgenes[randnums], all_families[randnums]


@ray.remote
def fastfindOrderNP_Nmin(file_name, seq_in_samples, cleaned_path, start_round, end_round):
    lengths = np.load(cleaned_path + "/lengths/" + file_name)
    families = np.load(cleaned_path + "/families/" + file_name)
    Jgenes = np.load(cleaned_path + "/Jgenes/" + file_name)
    Vgenes = np.load(cleaned_path + "/Vgenes/" + file_name)
    data = np.load(cleaned_path + '/cubes/' + file_name)

    end_round = max((len(lengths) * end_round) // seq_in_samples,1)

    for i in range(start_round, start_round + end_round):
        # for i in range(start_round,end_round):
        if(len(lengths) > seq_in_samples):
            randnums = np.random.choice(range(len(data)), seq_in_samples, replace=False)
        else:
            randnums = np.random.choice(range(len(data)), seq_in_samples, replace=True)
        sample = data[randnums]
        sample_lengths = lengths[randnums]
        sample_families = families[randnums]
        sample_Jgenes = Jgenes[randnums]
        sample_Vgenes = Vgenes[randnums]
        sorted_indices = np.lexsort((sample_Vgenes, sample_Jgenes, sample_families, sample_lengths))
        sorted_sample = sample[sorted_indices]
        sorted_lengths = sample_lengths[sorted_indices]
        sorted_families = sample_families[sorted_indices]
        sorted_Vgenes = sample_Vgenes[sorted_indices]
        sorted_Jgenes = sample_Jgenes[sorted_indices]

        np.save(cleaned_path + '/cubes/dataINorder/' + str(i) + '_' + file_name, sorted_sample)
        np.save(cleaned_path + '/cubes/lengthsINorder/' + str(i) + '_' + file_name, sorted_lengths)
        np.save(cleaned_path + '/cubes/familiesINorder/' + str(i) + '_' + file_name, sorted_families)
        np.save(cleaned_path + '/cubes/VgenesINorder/' + str(i) + '_' + file_name, sorted_Vgenes)
        np.save(cleaned_path + '/cubes/JgenesINorder/' + str(i) + '_' + file_name, sorted_Jgenes)
    print(file_name + '  ready, ' + str(end_round) + ' rounds')

    return True

@ray.remote
def fastfind_global_local_order(file_name, seq_in_samples, cleaned_path, start_round, end_round,
                                all_l_f_j_v_percent, num_of_clusters, frame_size=4):
    lengths = np.load(cleaned_path + "/lengths/" + file_name)
    families = np.load(cleaned_path + "/families/" + file_name)
    Jgenes = np.load(cleaned_path + "/Jgenes/" + file_name)
    Vgenes = np.load(cleaned_path + "/Vgenes/" + file_name)
    data = np.load(cleaned_path + '/cubes/' + file_name)

    end_round = max((len(lengths) * end_round) // seq_in_samples,1)

    for i in range(start_round, start_round + end_round):
        # for i in range(start_round,end_round):
        if(len(lengths) > seq_in_samples):
            randnums = np.random.choice(range(len(data)), seq_in_samples, replace=False)
        else:
            randnums = np.random.choice(range(len(data)), seq_in_samples, replace=True)
        sample = data[randnums]
        sample_Jgenes = Jgenes[randnums]
        sample_families = families[randnums]
        sample_Vgenes = Vgenes[randnums]
        sample_lengths = lengths[randnums]
        sorted_indices = np.lexsort((sample_lengths, sample_Vgenes, sample_families, sample_Jgenes))
        sorted_sample = sample[sorted_indices]
        sorted_Jgenes = sample_Jgenes[sorted_indices]
        sorted_families = sample_families[sorted_indices]
        sorted_Vgenes = sample_Vgenes[sorted_indices]
        sorted_lengths = sample_lengths[sorted_indices]

        all_l_f_j_v_percent_keys = list(all_l_f_j_v_percent.keys())
        final_sample = np.zeros((num_of_clusters * frame_size, 87, 4),dtype=np.int8)
        fake_seqs = np.zeros((num_of_clusters * frame_size, 10),dtype=np.float32) - 1.0
        global_order_indexes = np.zeros((num_of_clusters * frame_size),dtype=np.int8)
        j_, f_, v_, l_ = sorted_Jgenes[0], sorted_families[0], sorted_Vgenes[0], sorted_lengths[0]     # last step
        pointer1 = 0
        pointer2 = 0
        for i in range(len(sorted_Jgenes)):
            j, f, v, l = sorted_Jgenes[i], sorted_families[i], sorted_Vgenes[i], sorted_lengths[i]
            if (j_ == j) & (f_ == f) & (v_ == v) & (l_ == l):
                j_, f_, v_, l_ = j, f, v, l
                continue
            else:
                cluster_name = str(j_)+'_'+str(f_)+'_'+str(v_)+'_'+str(l_)

                if (cluster_name in all_l_f_j_v_percent_keys) & (cluster_name + '_extra' in all_l_f_j_v_percent_keys): # extra frame
                    num_of_seqs_in_cluster = min(frame_size*2, i - pointer1)
                    sub_random = np.random.choice(range(pointer1,i), num_of_seqs_in_cluster, replace=False)
                    first_frame_indexes = sub_random[0:num_of_seqs_in_cluster//2]
                    second_frame_indexes = sub_random[num_of_seqs_in_cluster//2:num_of_seqs_in_cluster]
                    position_in_global_order = all_l_f_j_v_percent_keys.index(cluster_name)
                    final_sample[position_in_global_order*frame_size:position_in_global_order*frame_size + num_of_seqs_in_cluster//2] = sorted_sample[first_frame_indexes]
                    final_sample[(position_in_global_order + 1)*frame_size:(position_in_global_order + 1)*frame_size + num_of_seqs_in_cluster - num_of_seqs_in_cluster//2] = sorted_sample[second_frame_indexes]
                    fake_seqs[position_in_global_order*frame_size:position_in_global_order*frame_size + num_of_seqs_in_cluster//2] = 0
                    fake_seqs[(position_in_global_order + 1)*frame_size:(position_in_global_order + 1)*frame_size + num_of_seqs_in_cluster - num_of_seqs_in_cluster//2] = 0
                    pointer1 = i
                    j_, f_, v_, l_ = j, f, v, l
                elif (cluster_name in all_l_f_j_v_percent_keys):
                    num_of_seqs_in_cluster = min(frame_size, i - pointer1)
                    sub_random = np.random.choice(range(pointer1, i), num_of_seqs_in_cluster, replace=False)
                    position_in_global_order = all_l_f_j_v_percent_keys.index(cluster_name)
                    final_sample[position_in_global_order * frame_size:position_in_global_order * frame_size + num_of_seqs_in_cluster] = sorted_sample[sub_random]
                    fake_seqs[position_in_global_order * frame_size:position_in_global_order * frame_size + num_of_seqs_in_cluster] = 0
                    pointer1 = i
                    j_, f_, v_, l_ = j, f, v, l
                else:
                    pointer1 = i
                    j_, f_, v_, l_ = j, f, v, l


        np.save(cleaned_path + '/cubes/dataINorder/' + str(i) + '_' + file_name, final_sample)
        np.save(cleaned_path + '/cubes/lengthsINorder/' + str(i) + '_' + file_name, sorted_lengths)
        np.save(cleaned_path + '/cubes/familiesINorder/' + str(i) + '_' + file_name, sorted_families)
        np.save(cleaned_path + '/cubes/VgenesINorder/' + str(i) + '_' + file_name, sorted_Vgenes)
        np.save(cleaned_path + '/cubes/JgenesINorder/' + str(i) + '_' + file_name, sorted_Jgenes)
    print(file_name + '  ready, ' + str(end_round) + ' rounds')

    return True


def rearange_data(args):
    # a1 = datetime.datetime.now()
    result = []
    # for fastfind_global_local_order #
    global_order = False
    removed, added = 0, 0
    if global_order:
        new_dict = {}
        for k in all_j_f_v_l_percent:
            if all_j_f_v_l_percent[k] < 0.000001:
                removed += 1
            elif all_j_f_v_l_percent[k] > 0.00001:
                new_dict[k] = all_j_f_v_l_percent[k]
                new_dict[k+'_extra'] = 0
                added += 1
            else:
                new_dict[k] = all_j_f_v_l_percent[k]

        num_of_clusters = len(new_dict)
        print('removed: ',removed)
        print('added: ',added)
        print('total: ',num_of_clusters)

    for file_name in os.listdir(args.cleaned_path + '/cubes/'):
        if (file_name.endswith('.npy')):
            # fastfindOrderNP_Nmin(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round)
            result.append(fastfindOrderNP_Nmin.remote(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round))
            # result.append(fastfind_global_local_order.remote(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round,new_dict,num_of_clusters))
            # result.append(fastfind_global_local_order(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round,new_dict, num_of_clusters))

    for r in result:
        ray.wait([r])
    # b1 = datetime.datetime.now()
    # print((b1 - a1).total_seconds())

default_cleaned_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/Cleaned'

if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--cleaned_path', help='cleaned_path', type=str, default=default_cleaned_path)
    parser.add_argument('--balance_data', type=bool, default=False, help='balance_data for arrange_from_cube')
    parser.add_argument('--seq_in_samples', type=int, default=32000, help='sequences in each random sample')
    parser.add_argument('--start_round', type=int, default=0, help='start_round for arrange_from_cube')
    parser.add_argument('--end_round', type=int, default=1, help='end_round for arrange_from_cube')
    parser.add_argument('--sleep_time', type=float, default=0.2, help='sleep_time for arrange_from_cube')
    args = parser.parse_args()

    # print("Number of processors: ", mp.cpu_count())
    print("start arrange_by_length")

    rearange_data(args)

    print("end arrange_by_length")

