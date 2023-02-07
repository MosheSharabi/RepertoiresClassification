# RepertoiresClassification
## Prepare training dataset out of raw repertoires (.csv)
* To simulate new data base on realworld samples run:
```shell
python3 simulate_data.py --cmv_path + args.cmv_path + --tsv_path + args.tsv_path
```
* To preprocess the data run:
```shell
python3 PrepareData.py --tsv_path + args.tsv_path + --dest_path + args.cleaned_path
```
* To split the dataset into K folds for the cross validation run:
```shell
python3 K_Fold.py --cleaned_path + args.cleaned_path
```
* To sub-sample the repertoires and reorder the sub-samples into 1D or 2D run:
for 1D:
```shell
python3 arrange_1D.py --cleaned_path + args.cleaned_path + --end_round + args.end_round +
              --sleep_time + args.sleep_time + --GIB_per_obj + args.GIB_per_obj + --ray_cpus + args.ray_cpus
```
for 2D:
```shell
python3 arrange_2D.py --cleaned_path + args.cleaned_path + --end_round + args.end_round +
              --sleep_time + args.sleep_time
```
