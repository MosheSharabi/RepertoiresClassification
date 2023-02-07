import argparse
import math
import time
import numpy as np
import multiprocessing as mp
from l_f_j_v_lists import all_j_f_v_l_percent
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
# ray.init(memory= 60000000000, object_store_memory=60000000000)
# ray.init()

# create a dictionary containing pairs of {j_f_v_l_key : [list of all indexes belonging to that key]}
def get_rep_j_f_v_l_indexes_dic(lengths, families, Jgenes, Vgenes,fixed_places_j_f_v_l_dic):
    rep_j_f_v_l_indexes_dic = {key: [] for key in fixed_places_j_f_v_l_dic}
    for i in range(len(lengths)):
        key = str(Jgenes[i]) + '_' + str(families[i]) + '_' + str(Vgenes[i]) + '_' + str(lengths[i])
        if key in rep_j_f_v_l_indexes_dic:
            rep_j_f_v_l_indexes_dic[key].append(i)
            # rep_j_f_v_l_indexes_dic[key] = rep_j_f_v_l_indexes_dic[key].append(i)
    return rep_j_f_v_l_indexes_dic

@ray.remote
def fastfindOrderNP_Nmin(file_name, seq_in_samples, cleaned_path, start_round, end_round,fixed_places_j_f_v_l_dic):
    Jgenes = np.load(cleaned_path + "/Jgenes/" + file_name)
    families = np.load(cleaned_path + "/families/" + file_name)
    Vgenes = np.load(cleaned_path + "/Vgenes/" + file_name)
    lengths = np.load(cleaned_path + "/lengths/" + file_name)
    data = np.load(cleaned_path + '/cubes/' + file_name)
    # add null to the end
    Jgenes = np.append(Jgenes, np.zeros((1),dtype=np.int8))
    families = np.append(families, np.zeros((1),dtype=np.int8))
    Vgenes = np.append(Vgenes, np.zeros((1),dtype=np.int8))
    lengths = np.append(lengths, np.zeros((1),dtype=np.int8))
    data = np.append(data, np.zeros((1,87,4),dtype=np.int8),axis=0)


    end_round = max((len(lengths) * end_round) // seq_in_samples,1)

    rep_j_f_v_l_indexes_dic = get_rep_j_f_v_l_indexes_dic(lengths, families, Jgenes, Vgenes, fixed_places_j_f_v_l_dic)

    for i in range(start_round, start_round + end_round):
        # for i in range(start_round,end_round):
        randnums = np.empty(0,dtype=int)
        for key in fixed_places_j_f_v_l_dic:
            if(len(rep_j_f_v_l_indexes_dic[key])==0):
                randnums_in_clone = np.zeros(fixed_places_j_f_v_l_dic[key], dtype=int) -1
            elif(len(rep_j_f_v_l_indexes_dic[key]) >= fixed_places_j_f_v_l_dic[key]):
                randnums_in_clone = np.random.choice(rep_j_f_v_l_indexes_dic[key], fixed_places_j_f_v_l_dic[key], replace=False)
            else:
                randnums_in_clone = np.random.choice(rep_j_f_v_l_indexes_dic[key], fixed_places_j_f_v_l_dic[key], replace=True)

            randnums  = np.append(randnums, randnums_in_clone)

        sample = data[randnums]
        sample_lengths = lengths[randnums]
        sample_families = families[randnums]
        sample_Jgenes = Jgenes[randnums]
        sample_Vgenes = Vgenes[randnums]
        # print(sum([1 if i == 0 else 0 for i in sample_lengths]))
        np.save(cleaned_path + '/cubes/dataINorder/' + str(i) + '_' + file_name, sample)
        np.save(cleaned_path + '/cubes/lengthsINorder/' + str(i) + '_' + file_name, sample_lengths)
        np.save(cleaned_path + '/cubes/familiesINorder/' + str(i) + '_' + file_name, sample_families)
        np.save(cleaned_path + '/cubes/VgenesINorder/' + str(i) + '_' + file_name, sample_Vgenes)
        np.save(cleaned_path + '/cubes/JgenesINorder/' + str(i) + '_' + file_name, sample_Jgenes)
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

def get_fixed_places_j_f_v_l_dic(args):
    average_sizes_dic = np.load(os.getcwd() +'/average_sizes_dic.npy', allow_pickle='TRUE').item()  # containing averages of the relative clones sizes in their own repertoires
    s = np.sum([average_sizes_dic[k] for k in average_sizes_dic]) # will Not be sum to total of 1
    normalized_average_sizes_dic = {k: average_sizes_dic[k]/s for k in average_sizes_dic} # will be sum to total of 1
    fixed_places_j_f_v_l_dic = {k: math.floor((args.seq_in_samples * normalized_average_sizes_dic[k])**0.9) if
                    (args.seq_in_samples * normalized_average_sizes_dic[k]) >= 1 else 1 for k in normalized_average_sizes_dic}
    # sort indexes
    j_f_v_l_list = [k.split('_') for k in fixed_places_j_f_v_l_dic]
    j_f_v_l_list = np.array(j_f_v_l_list).astype(np.int8)
    sorted_indices = np.lexsort((j_f_v_l_list[:, 3], j_f_v_l_list[:, 2], j_f_v_l_list[:, 1], j_f_v_l_list[:, 0]))
    s = j_f_v_l_list[sorted_indices]
    sorted_fixed_places_j_f_v_l_dic = {str(s[i,0])+'_'+str(s[i,1])+'_'+str(s[i,2])+'_'+str(s[i,3]) :
             fixed_places_j_f_v_l_dic[str(s[i,0])+'_'+str(s[i,1])+'_'+str(s[i,2])+'_'+str(s[i,3])] for i in range(len(s))}
    return sorted_fixed_places_j_f_v_l_dic, sum([sorted_fixed_places_j_f_v_l_dic[k] for k in sorted_fixed_places_j_f_v_l_dic])

def rearange_data(args):
    # a1 = datetime.datetime.now()
    result = []
    fixed_places_j_f_v_l_dic, final_seq_in_sample = get_fixed_places_j_f_v_l_dic(args)
    print('final_seq_in_sample: ', final_seq_in_sample)
    for file_name in os.listdir(args.cleaned_path + '/cubes/'):
        if (file_name.endswith('.npy')):
            # fastfindOrderNP_Nmin(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round,fixed_places_j_f_v_l_dic)
            result.append(fastfindOrderNP_Nmin.remote(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round,fixed_places_j_f_v_l_dic))
            # result.append(fastfind_global_local_order.remote(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round,new_dict,num_of_clusters))
            # result.append(fastfind_global_local_order(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round,new_dict, num_of_clusters))
            time.sleep(args.sleep_time)

    for r in result:
        ray.wait([r])
    # b1 = datetime.datetime.now()
    # print((b1 - a1).total_seconds())

default_cleaned_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/cleaned'

if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--cleaned_path', help='cleaned_path', type=str, default=default_cleaned_path)
    parser.add_argument('--balance_data', type=bool, default=False, help='balance_data for arrange_from_cube')
    parser.add_argument('--seq_in_samples', type=int, default=32000, help='sequences in each random sample')
    parser.add_argument('--start_round', type=int, default=0, help='start_round for arrange_from_cube')
    parser.add_argument('--end_round', type=int, default=5, help='end_round for arrange_from_cube')
    parser.add_argument('--sleep_time', type=float, default=1)
    parser.add_argument('--GIB_per_obj', type=int, default=1)
    parser.add_argument('--ray_cpus', type=int, default=4)
    args = parser.parse_args()

    ray.init(object_store_memory=args.GIB_per_obj * 1024 * 1024 * 1024, num_cpus=args.ray_cpus)

    print("start arrange_1D")

    rearange_data(args)

    print("end arrange_1D")

