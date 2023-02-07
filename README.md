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
* To balance the folds to have equal number of positive and negative sub-sample run:
```shell
python3 balance_folds.py --cubes_path + args.cleaned_path + 'cubes/'
```

## Model training
* To train the network to classify between positive and negative repertoiers run:
```shell
python3 main_classifier_new.py --model Classifier_new  --cleaned_path /home/mnt/dsi_vol1/shared/moshe/datasets/cmv/Cleaned_32000_87_templates_folds
--val_folds f1 --train_folds f0 f2 f3 f4 --max_epoch 14 --batch_size 4 --seq_in_samples 32000 --save_dir 01 --dataset cmv --model_type 21
--sub_model_type 3 --device /gpu:0 --steps_per_epoch 200 --dropout 0.2 --initial_LR 0.001 --top_k 100 --decay 0.8 --seed 1
```
