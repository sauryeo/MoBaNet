import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    # parser.add_argument('-encoder', type=str, default='dinov2_vitg14', help='encoder type: sam_vit_b/sam_vit_l/sam_vit_h/dinov2_vits14/dinov2_vitb14/dinov2_vitl14/dinov2_vitg14')
    parser.add_argument('-encoder', type=str, default='dinov2_vitl14', help='encoder type: sam_vit_b/sam_vit_l/sam_vit_h/dinov2_vits14/dinov2_vitb14/dinov2_vitl14/dinov2_vitg14')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    # parser.add_argument('-mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument('-mod', type=str, default='sam_peft', help='mod type:seg,cls,val_ad')
    parser.add_argument('-exp_name', default='', type=str, help='experiment name used for logging/checkpoints')
    parser.add_argument('-mode', type=str, default='Test', help='running mode: Train or Test')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=int, default=None, help='visualization')
    parser.add_argument('-reverse', type=str2bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=str2bool, default=False, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=5,help='interval between each validation')
    parser.add_argument('-gpu', type=str2bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=266, help='image_size (set to padded patch size for DINOv2)')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=1, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=2, help='batch size for dataloader')
    parser.add_argument('-s', type=str2bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    # parser.add_argument('-weights', type=str, default = '/data1/lihaocheng/deeplearning/paper1/dinov2/resultsp/Net_epoch35_0.873357399055655', help='the weights file you want to test')
    parser.add_argument('-weights', type=str, default = '/data1/lihaocheng/deeplearning/paper1/MoBaSF-dinov2/resultsp/Net_epoch35_0.8792773319039509', help='the weights file you want to test')
    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='isic' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', default=None , help='sam checkpoint address')
    # parser.add_argument('-dinov2_ckpt', default='/data1/lihaocheng/deeplearning/paper1/MoBaSF-dinov2/weights/dinov2_vitg14_pretrain.pth' , help='dinov2 checkpoint file or folder')
    parser.add_argument('-dinov2_ckpt', default='/data1/lihaocheng/deeplearning/paper1/MoBaSF-dinov2/weights/dinov2_vitl14_pretrain.pth' , help='dinov2 checkpoint file or folder')
    parser.add_argument('-dinov2_hub_dir', default=os.getenv("DINOV2_HUB_DIR", "dinov2_hub") , help='local dinov2 repo path for torch.hub')
    parser.add_argument('-dinov2_strict', type=str2bool, default=False , help='require strict DINOv2 weight loading')
    parser.add_argument('-dinov2_use_extra_patch_embed', type=str2bool, default=True , help='use trainable extra patch_embed for auxiliary modality in DINOv2')
    parser.add_argument('-dinov2_tap_indices', type=str, default='' , help='comma-separated DINOv2 tap indices, e.g. 5,11,17,23')
    parser.add_argument('-thd', type=str2bool, default=False , help='3d or not')
    parser.add_argument('-chunk', type=int, default=None , help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
    parser.add_argument('-evl_chunk', type=int, default=None , help='evaluation chunk')
    parser.add_argument('-mid_dim', type=int, default=None , help='middle dim of adapter or the rank of lora matrix')
    parser.add_argument('-peft_ratio', type=float, default=0.25, help='peft prompt ratio')
    parser.add_argument('-peft_adapter_ratio', type=float, default=0.0625, help='peft adapter ratio')
    parser.add_argument('-use_cpia', type=str2bool, default=True, help='enable CPIA blocks in the DINOv2 encoder')
    parser.add_argument('-use_dgfm', type=str2bool, default=True, help='enable dgfm blocks in the DINOv2 encoder')
    parser.add_argument('-mcrc', type=str2bool, default=True, help='enable MCRC local cross-modality region corruption')
    parser.add_argument('-mcrc_ratio', type=float, default=0.25, help='ratio of batch samples selected for MCRC')
    parser.add_argument('-mcrc_local_blocks', type=int, default=1, help='number of local masked blocks per selected sample')
    parser.add_argument('-mcrc_block_scale_min', type=float, default=0.1, help='minimum masked area ratio for each local block')
    parser.add_argument('-mcrc_block_scale_max', type=float, default=0.3, help='maximum masked area ratio for each local block')
    parser.add_argument('-mcrc_block_aspect_min', type=float, default=0.5, help='minimum aspect ratio for each local block')
    parser.add_argument('-mcrc_block_aspect_max', type=float, default=2.0, help='maximum aspect ratio for each local block')
    parser.add_argument('-mcrc_aux_weight', type=float, default=0.01, help='aux loss weight for MCRC')
    parser.add_argument('-eval_modalities', type=str, default='rgbd', help='comma-separated inference modalities: rgbd,rgb,dsm')
    parser.add_argument('-save_inference_maps', type=str2bool, default=True, help='save colorized inference maps in Test mode')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation, set 2 for REFUGE dataset.')
    parser.add_argument(
    '-data_path',
    type=str,
    default='../data',
    help='The path of segmentation data')
    # '../dataset/RIGA/DiscRegion'
    # '../dataset/ISIC'
    opt = parser.parse_args()

    return opt

# required=True, 
