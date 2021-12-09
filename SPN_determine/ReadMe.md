The training and testing commonds under deterministic inpainting setup is similiar with those used in SPL. Under this setup, we conduct experiments on Places2, CelebA, and Paris StreetView datasets.

There are currently some troubles with my google drive to share links, so I provide pretrained model on [BaiDu](https://pan.baidu.com/s/1Jd_lw6so5QjRcm1-K9RkdQ), code:2x99

Links on google drive will be updated if it works.

## Training
We use the DistributedDataParallel (DDP) to train our model, which means that for now, you need at least two GPU cards for training our model. To train our model with only one GPU, you need modify the initialization, datasetloader and optimization parts and I will provide a new version in the future. 

Take the Paris dataset for example, the training commond is as follows:  
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
        --bs 4 --gpus 2 --prefix SPL_paris --with_test \
        --img_flist your/train/flist/of/paris --mask_flist your/flist/of/mask --test_img_flist your/test/flist/of/paris \
        --test_mask_flist your/flist/of/masks --test_mask_index your/npy/file/to/form/img-mask/pairs \
        --dataset paris  --TRresNet_path path/of/ASL/weight --nEpochs 70
```
If you want to retrain your model, you need add 
```bash
--pretrained True --pretrained_sr checkpoints/of/your/model --start_epoch 4
```
During our training stage, we use --test_img_flist and --test_mask_index to evaluate the performance of current model. You can change the evaluation number with parameter --val_prob_num or directly remove the parameter --with_test, in which case only the latest model weights will be saved after each epoch.

For Paris dataset, we train our model for 75 epochs and we deacy the learning rete at about 50 epochs with 0.1. Besides, in the last 15 epochs we remove the prior reconstruction loss as we find this can further improve the performance. 

## Test and Evaluation
The evaluation commond is as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_final.py --bs 50 --gpus 1 --dataset paris \
        --img_flist your/test/image/flist/ --mask_flist your/flist/of/masks --mask_index your/npy/file/to/form/img-mask/pairs \
        --model checkpoints/x_launcherRN_bs_4_epoch_best.pt --save --save_path ./test_results
```
1) If you cannot successfully install the inplace-abn module, you can comment the ninth line (from src.models import create_model) in [models_inpaint.py](models_inpaint.py), the ASL model will not be established and you can still evaluate our model.
2) This commond will print the average PSNR, SSIM and L1 results and also save the predicted results in --save_path. You can remove the paramter --save and no images will be saved.
3) For FID score, we fisrt generate all restored images and use the code from [PICNet](https://github.com/lyndonzheng/Pluralistic-Inpainting).
