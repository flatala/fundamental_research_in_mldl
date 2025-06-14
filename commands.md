## Baseline training / fine-tuning command

python SDN/train.py \
--video_path <path_to_dir_with_JPG_dataset_subdirs> \
--annotation_path <data_split_json_path> \
--result_path <your_result_dir_path> \
--root_path <your_root_dir_path> \
--pretrain_path <your_pretrianed_checkpoint_path> \
--dataset ucf101 \
--n_classes 101 \
--n_finetune_classes 101 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--batch_size 256 \
--val_batch_size 16 \
--n_threads 16 \
--checkpoint 1 \
--ft_begin_index 0 \
--is_mask_adv \
--learning_rate 0.0032 \
--weight_decay 1e-5 \
--n_epochs 100 


## Scene debiasing fine-tuning command


python SDN/train.py \
--video_path <path_to_dir_with_JPG_dataset_subdirs> \
--annotation_path <data_split_json_path> \
--result_path <your_result_dir_path> \
--root_path <your_root_dir_path> \
--place_pred_path <scene_label_directory> \
--pretrain_path <your_pretrianed_checkpoint_path> \
--dataset ucf101_adv \
--n_classes 101 \
--n_finetune_classes 101 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--batch_size 256 \
--val_batch_size 16 \
--n_threads 16 \
--checkpoint 1 \
--ft_begin_index 0 \
--num_place_hidden_layers 3 \
--new_layer_lr 1e-2 \
--learning_rate 0.0032 \
--weight_decay 1e-5 \
--warm_up_epochs 5 \
--n_epochs 80 \
--is_place_adv \
--is_place_entropy \
--is_entropy_max \
--alpha 1.0 \
--is_mask_adv \
--num_places_classes 365 


## Human masking + scene debiasing fine-tuning command


python train.py \
--video_path <path_to_dir_with_JPG_dataset_subdirs> \
--annotation_path <data_split_json_path> \
--result_path <your_result_dir_path> \
--root_path <your_root_dir_path> \
--place_pred_path <scene_label_directory> \
--pretrain_path <your_pretrianed_checkpoint_path> \
--human_dets_path <directory_with_train_and_val_mask_npy_files> \
--dataset ucf101 \
--n_classes 101 \
--n_finetune_classes 101 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--batch_size 256 \
--val_batch_size 16 \
--n_threads 16 \
--checkpoint 1 \
--ft_begin_index 0 \
--num_place_hidden_layers 3 \
--num_human_mask_adv_hidden_layers 1 \
--new_layer_lr 1e-4 \
--learning_rate 1e-4 \
--warm_up_epochs 0 \
--weight_decay 1e-5 \
--n_epochs 100 \
--is_place_entropy \
--is_entropy_max \
--is_mask_entropy \
--alpha 0.5 \
--mask_ratio 1.0 \
--slower_place_mlp \
--not_replace_last_fc \
--num_places_classes 365 

## Generating test annotation files / running inference

To run tetsing, append following to any of the training commands:

--test \
--no_train \
--no_val \
--test_subset <test/val> \
--adv_test_type <scene/action> 



## Calculating action classification accuracy from test run annotation file

python -m utils.eval_ucf101 \
    --annotation_path <data_split_json_path> \
    --prediction_path <val/tets.json_prediction_file_path>




























