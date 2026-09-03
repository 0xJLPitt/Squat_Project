r"""
目的: 執行 MotionAGFormer 模型，將 2D Human3.6M 骨架點提升為 3D 骨架點。
執行範例:
python infer_agformer.py
"""
import sys
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# 將 MotionAGFormer 路徑加入，方便載入模組
sys.path.append(r'D:\Pitt\Project\tools\MotionAGFormer')

from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer
import copy

def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result

def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    downsample = np.arange(243) # 預設 downsample
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
                downsample = np.arange(243)
    return clips, downsample

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data

def get_video_resolution(demo_dir, npz_path):
    """
    從 npz_path 所在的資料夾中尋找影片檔，讀取真實解析度。
    這對於垂直/直立式影片(480x640)特別重要，因為比例錯誤會導致3D轉換嚴重失真。
    """
    folder = os.path.dirname(npz_path)
    video_files = glob.glob(os.path.join(folder, '*.avi')) + glob.glob(os.path.join(folder, '*.mp4'))
    if len(video_files) > 0:
        cap = cv2.VideoCapture(video_files[0])
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if w > 0 and h > 0:
            # 為了避免直式影片(480x640)的Y座標在正規化時超出 [-1, 1] 導致模型崩潰，
            # 我們將畫面視為 max(w,h) x max(w,h) 的正方形，這樣座標就會乖乖留在 [-1, 1] 內！
            size = int(max(w, h))
            return size, size
    
    # 預設值 (如果找不到影片)
    return 640, 640

def infer_3d(npz_file, output_file, model, w, h):
    # 讀取剛剛轉換好的 Human3.6M 17 關節點格式 (1, N_frames, 17, 3)
    data = np.load(npz_file, allow_pickle=True)
    keypoints = data['reconstruction']
    
    # 這裡只取前兩個維度 [x, y]，因為 MotionAGFormer 不需要 confidence (或者只使用 x, y)
    # 不過 normalize_screen_coordinates 可以接受最後維度為 3 (會一併保留 conf)
    
    # 將序列分段為 243 幀 (MotionAGFormer 要求的長度)
    clips, downsample = turn_into_clips(keypoints)

    output_3D_all = []

    for idx, clip in enumerate(clips):
        # 將 x,y 座標作正規化
        input_2D = normalize_screen_coordinates(clip, w=w, h=h) 
        input_2D_aug = flip_data(input_2D)
        
        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        # 進行 3D 推論
        with torch.no_grad():
            output_3D_non_flip = model(input_2D) 
            output_3D_flip = flip_data(model(input_2D_aug))
            output_3D = (output_3D_non_flip + output_3D_flip) / 2

        # 處理尾段重複的幀
        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        # 將 pelvis (根節點) 對齊到原點
        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()
        
        # 轉換座標系 (camera_to_world)
        for j in range(len(post_out_all)):
            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out_all[j] = camera_to_world(post_out_all[j], R=rot, t=0)
            post_out_all[j, :, 2] -= np.min(post_out_all[j, :, 2])
            
        output_3D_all.append(post_out_all)

    # 最終輸出的 3D 骨架，shape: (N_frames, 17, 3)
    final_3D = np.concatenate(output_3D_all, axis=0)
    
    np.savez_compressed(output_file, reconstruction=final_3D)
    print(f"-> 轉換成功！3D 結果儲存至: {output_file}")


def load_model(checkpoint_path):
    class Args: pass
    args = Args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243

    model = nn.DataParallel(MotionAGFormer(**vars(args))).cuda()
    
    pre_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()
    return model

# Compatibility aliases
load_agformer_model = load_model
infer_agformer_3d = infer_3d

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Infer MotionAGFormer 3D Pose from 2D Human3.6M npz")
    parser.add_argument('--npz', type=str, default=None, help="Path to single keypoints.npz")
    parser.add_argument('--out', type=str, default=None, help="Path to output keypoints_3d.npz")
    parser.add_argument('--dir', type=str, default=None, help="Batch directory containing keypoints.npz")
    parser.add_argument('--ckpt', type=str, default=r'D:\Pitt\Project\tools\MotionAGFormer\checkpoint\motionagformer-b-h36m.pth.tr', help="Path to MotionAGFormer checkpoint")
    parser.add_argument('--width', type=int, default=None, help="Video width for normalization (optional)")
    parser.add_argument('--height', type=int, default=None, help="Video height for normalization (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"錯誤：找不到模型權重 {args.ckpt}")
        print("請參考 MotionAGFormer 的 README，下載 'MotionAGFormer-B' H3.6M weights 到 checkpoint 資料夾。")
        sys.exit(1)
        
    print(f"載入 MotionAGFormer 預訓練模型: {args.ckpt} ...")
    model = load_model(args.ckpt)
    
    if args.npz:
        out_npz = args.out if args.out else args.npz.replace('keypoints.npz', 'keypoints_3d.npz')
        if args.width and args.height:
            w, h = args.width, args.height
        else:
            w, h = get_video_resolution(os.path.dirname(args.npz), args.npz)
        print(f"推論單一檔案: {args.npz} -> {out_npz} (解析度: {w}x{h})")
        infer_3d(args.npz, out_npz, model, w=w, h=h)
    else:
        target_dir = args.dir if args.dir else r'D:\Pitt\Project\squat\demo'
        print(f"開始掃描資料夾: {target_dir}")
        npz_files = glob.glob(os.path.join(target_dir, '**', 'keypoints.npz'), recursive=True)
        if not npz_files:
            print(f"在 {target_dir} 中找不到任何 keypoints.npz 檔案。請先執行 convert_keypoints.py")
            sys.exit(0)
        
        for npz in npz_files:
            out_npz = npz.replace('keypoints.npz', 'keypoints_3d.npz')
            print(f"\n處理 2D 骨架檔案: {npz}")
            w, h = get_video_resolution(target_dir, npz)
            infer_3d(npz, out_npz, model, w=w, h=h)
