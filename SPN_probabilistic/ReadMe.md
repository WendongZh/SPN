The training and testing commonds under probabilistic inpainting setup is similiar with those used in SPL. Under this setup, we conduct experiments on CelebA-HQ dataset.

I'm sorry about that this part includes some redundant files such as bilinear.py and models (actually, this directory contains codes for yolov5 and we use it in our ablation study part.)

There are currently some troubles with my google drive to share links, so I provide pretrained model on [BaiDu](https://pan.baidu.com/s/1kgcs9_LFRoA40F81h_rHVQ), code:hqek

Links on google drive will be updated if it works.

## Training
We use the DistributedDataParallel (DDP) to train our model, which means that for now, you need at least two GPU cards for training our model. To train our model with only one GPU, you need modify the initialization, datasetloader and optimization parts. 

Take the celeba-hq dataset for example, the training commond is as follows:  
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
        --bs 4 --gpus 2 --prefix SPL_celeba-hq --with_test \
        --img_flist your/train/flist/ --mask_flist your/flist/of/mask --test_img_flist your/test/flist/ \
        --test_mask_flist your/flist/of/masks --test_mask_index your/npy/file/to/form/img-mask/ \
        --dataset celeba-hq  --TRresNet_path path/of/ASL/weight --nEpochs 70
```
If you want to retrain your model, you need add 
```bash
--pretrained True --pretrained_sr checkpoints/of/your/model --start_epoch 4
```
During our training stage, we use --test_img_flist and --test_mask_index to evaluate the performance of current model. You can change the evaluation number with parameter --val_prob_num or directly remove the parameter --with_test, in which case only the latest model weights will be saved after each epoch.

## Test and Evaluation
The evaluation commond is as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_final_savenp_max.py --bs 50 --gpus 1 --dataset celeba-hq \
        --img_flist your/test/image/flist/ --mask_flist your/flist/of/masks --mask_index your/npy/file/to/form/img-mask/ \
        --model checkpoints/x_launcherRN_bs_4_epoch_best.pt --save --save_path ./test_results
```
1) If you cannot successfully install the inplace-abn module, you can comment the eleventh line (from src.models import create_model) in [models_inpaint.py](models_inpaint.py), the ASL model will not be established and you can still evaluate our model.
2) As we mentioned in our paper, this commond will random sample 5 results for each input and calculate the average MAX PSNR, SSIM and L1 results. All 5 results will be saved in --save_path. You can remove the paramter --save and no images will be saved.
3) For FID score, we fisrt generate all restored images and use the code from [PICNet](https://github.com/lyndonzheng/Pluralistic-Inpainting).
