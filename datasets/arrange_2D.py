import argparse
import numpy as np
import multiprocessing as mp
from l_f_j_v_lists import all_j_f_v_l_percent
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
# ray.init()
# ray.init(memory= 60000000000, object_store_memory=60000000000)

def get_distinct_Vgenes(Vfamilies,Vgenes):
    return Vfamilies.astype(np.int16) * 10 + Vgenes

def fastindexMask3D(arr, mask1 , mask2):
    l = list(mask1.transpose())
    newArr = arr[tuple(l)]
    newArr = np.take(newArr, mask2, axis=1)
    return newArr

def findOrder_2D(sample, lengths, Vgenes, Jgenes, families, seq_in_sample):
    dim = int(np.sqrt(seq_in_sample))
    start_seq = np.array([7,6,65,42]) # Jgene, family, Vgene, length
    x = np.concatenate(([Jgenes], [families], [Vgenes], [lengths]), axis=0)

    num_of_seq = seq_in_sample
    # A: contains the new arrangement, Initialized -1 to indicate square is empty
    A = np.zeros(shape=(dim, dim), dtype=int) - 1
    # B: contains the distance vectors, Initialized 100 to be far from chosen as minimum
    B = np.zeros(shape=(dim, dim, seq_in_sample), dtype=float) + 100
    # C: Contains the number neighbors, Initialized 0 as there is no neighbors in the beginning
    C = np.zeros(shape=(dim, dim), dtype=int)

    # mask1 = np.arange(0)   # holds vectors indices we should search in an argmin in B
    mask1 = np.array([np.zeros(2)], np.int32)  # bug
    mask1 = np.delete(mask1, 0, axis=0)  # bug
    mask2 = np.arange(seq_in_sample)  # holds vectors indices we didn't yet position in A

    # place random vector matching the start_seq metadata
    start_seq_dummy_index = np.argmax((x[0] == start_seq[0]) & (x[1] == start_seq[1]) & (x[2] == start_seq[2]) & (x[3] == start_seq[3]))
    # remove start_seq_dummy_index
    mask2 = mask2[mask2 != start_seq_dummy_index]
    A[dim // 2, dim // 2] = start_seq_dummy_index
    num_of_seq -= 1
    # DistanceFromAllVectors (in mask2)
    disVec = np.sum(np.bitwise_not(np.concatenate(([x[0][mask2] == start_seq[0]] , [x[1][mask2] == start_seq[1]] , [x[2][mask2] == start_seq[2]] , [x[3][mask2] == start_seq[3]]),axis=0)),axis=0)
    # disVec = np.mean(np.square(np.subtract(x[mask2], tempVec)), axis=1)  # DistanceFromAllVectors (in mask2)
    # disVec[tempIndex] = 100  # no need to look at this vector anywhere again
    for j in range(dim // 2 - 1, dim // 2 + 2):
        for k in range(dim // 2 - 1, dim // 2 + 2):
            if (0 <= j < dim) & (0 <= k < dim):
                if (A[j, k] == -1):
                    if (C[j, k] == 0):
                        mask1 = np.concatenate((mask1, [[j, k]]), axis=0)
                    C[j, k] += 1
                    B[j, k][mask2] = disVec


    # go for all the rest vectors
    isend = 0
    # for i in range((num_of_seq - 1)//K):
    while (isend == 0):
        column = 0
        mask1_placed_list = []
        mask2_placed_list = []
        smallB = fastindexMask3D(B, mask1, mask2)

        K = 2
        K = max(smallB.shape[0] // 20, K)
        if (smallB.shape[0] <= K):
            K = smallB.shape[0]
        if (K == num_of_seq):
            isend = 1
        num_vec_to_place = K
        # print(K)

        all_K_min_in_rows = np.argpartition(smallB, num_vec_to_place - isend, axis=1)[:, :num_vec_to_place]
        all_K_min_in_value = smallB[np.arange(smallB.shape[0])[:, None], all_K_min_in_rows]  # value = index_2

        while (num_vec_to_place):
            sorted_rows = np.argsort(all_K_min_in_value[:, column])
            column += 1
            save_for_later_list = []
            for i_mask_idx_0, mask_idx_0 in enumerate(sorted_rows):
                mask_idx_1 = all_K_min_in_rows[mask_idx_0][column - 1]
                if ((mask_idx_0 in mask1_placed_list) | (mask_idx_1 in mask2_placed_list)):
                    save_for_later_list.append(mask_idx_0)
                    continue
                    # TODO check how much time it happened
                else:
                    mask1_placed_list.append(mask_idx_0)
                    mask2_placed_list.append(mask_idx_1)
                    num_vec_to_place -= 1
                    num_of_seq -= 1

                    [idx_0, idx_1] = mask1[mask_idx_0]  # convert mask_idx to real indeces in A/B/C (mask1)
                    idx_2 = mask2[mask_idx_1]  # convert mask_idx to real indeces in x (mask2)

                    A[idx_0, idx_1] = idx_2

                    tempVec = x[:,idx_2]
                    disVec = np.sum(np.bitwise_not(np.concatenate(([x[0][mask2] == tempVec[0]], [x[1][mask2] == tempVec[1]],
                                                    [x[2][mask2] == tempVec[2]], [x[3][mask2] == tempVec[3]]),
                                                   axis=0)), axis=0)

                    # update surrounding squares in B
                    for j in range(idx_0 - 1, idx_0 + 2):
                        for k in range(idx_1 - 1, idx_1 + 2):
                            # if (0 <= j < dim) & (0 <= k < dim):
                            # make smooth ends
                            if (j == dim):
                                j = 0
                            if (k == dim):
                                k = 0
                            if (A[j, k] == -1):
                                if (C[j, k] == 0):
                                    mask1 = np.concatenate((mask1, [[j, k]]), axis=0)
                                B[j, k][mask2] = (C[j, k] * B[j, k][mask2] + disVec) / (
                                        C[j, k] + 1)  # update B with relative weight
                                C[j, k] += 1


                if (num_vec_to_place == 0):
                    break
                if (num_of_seq == 0):
                    break
        # b3 = datetime.datetime.now()
        # delta3 += (b3 - a3).total_seconds()
        mask1 = np.delete(mask1, mask1_placed_list, axis=0)  # update mask1
        mask2 = np.delete(mask2, mask2_placed_list, axis=0)  # update mask2

    reorder_sample = sample[A]
    reorder_lengths = lengths[A]
    reorder_Vgenes = Vgenes[A]
    reorder_Jgenes = Jgenes[A]
    reorder_families = families[A]

    return reorder_sample, reorder_Jgenes, reorder_families, reorder_Vgenes, reorder_lengths

# @ray.remote
def findOrder_2D_fast(file_name, seq_in_sample, cleaned_path, start_round, end_round):
    lengths = np.load(cleaned_path + "/lengths/" + file_name)
    families = np.load(cleaned_path + "/families/" + file_name)
    Jgenes = np.load(cleaned_path + "/Jgenes/" + file_name)
    Vgenes = np.load(cleaned_path + "/Vgenes/" + file_name)
    data = np.load(cleaned_path + '/cubes/' + file_name)

    end_round = max((len(lengths) * end_round) // seq_in_sample,1)

    for i in range(start_round, start_round + end_round):
        # for i in range(start_round,end_round):
        if(len(lengths) > seq_in_sample):
            randnums = np.random.choice(range(len(data)), seq_in_sample, replace=False)
        else:
            randnums = np.random.choice(range(len(data)), seq_in_sample, replace=True)
        sample = data[randnums]
        sample_lengths = lengths[randnums]
        sample_families = families[randnums]
        sample_Jgenes = Jgenes[randnums]
        sample_Vgenes = Vgenes[randnums]
        sample_Vgenes = get_distinct_Vgenes(sample_families, sample_Vgenes) # make each vgene unique

        sorted_sample, sorted_Jgenes, sorted_families, sorted_Vgenes, sorted_lengths =\
            findOrder_2D(sample, sample_lengths, sample_Vgenes, sample_Jgenes, sample_families, seq_in_sample)

        np.save(cleaned_path + '/cubes/dataINorder/' + str(i) + '_' + file_name, sorted_sample)
        np.save(cleaned_path + '/cubes/lengthsINorder/' + str(i) + '_' + file_name, sorted_lengths)
        np.save(cleaned_path + '/cubes/familiesINorder/' + str(i) + '_' + file_name, sorted_families)
        np.save(cleaned_path + '/cubes/VgenesINorder/' + str(i) + '_' + file_name, sorted_Vgenes)
        np.save(cleaned_path + '/cubes/JgenesINorder/' + str(i) + '_' + file_name, sorted_Jgenes)
    print(file_name + '  ready, ' + str(end_round) + ' rounds')

    return True


def rearange_data(args):
    # a1 = datetime.datetime.now()
    result = []

    for file_name in os.listdir(args.cleaned_path + '/cubes/'):
        if (file_name.endswith('.npy')):
            findOrder_2D_fast(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round)
            # result.append(findOrder_2D_fast.remote(file_name, args.seq_in_samples, args.cleaned_path, args.start_round, args.end_round))
            time.sleep(args.sleep_time)

    for r in result:
        ray.wait([r])
    # b1 = datetime.datetime.now()
    # print((b1 - a1).total_seconds())

default_cleaned_path = r'/home/moshe/Desktop/sequence_distance/cleaned'

if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--cleaned_path', help='cleaned_path', type=str, default=default_cleaned_path)
    parser.add_argument('--balance_data', type=bool, default=False, help='balance_data for arrange_from_cube')
    parser.add_argument('--seq_in_samples', type=int, default=22500, help='sequences in each random sample')
    parser.add_argument('--start_round', type=int, default=0, help='start_round for arrange_from_cube')
    parser.add_argument('--end_round', type=int, default=1, help='end_round for arrange_from_cube')
    parser.add_argument('--sleep_time', type=float, default=1)
    parser.add_argument('--GIB_per_obj', type=int, default=1)
    parser.add_argument('--ray_cpus', type=int, default=4)
    args = parser.parse_args()

    ray.init(object_store_memory=args.GIB_per_obj * 1024 * 1024 * 1024, num_cpus=args.ray_cpus)
    # print("Number of processors: ", mp.cpu_count())
    print("start arrange_2D")

    rearange_data(args)

    print("end arrange_2D")

