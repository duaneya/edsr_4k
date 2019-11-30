# coding=utf-8
# 测试集自动化处理脚本，注意该脚本默认在src目录下运行。对应工作路径要注意

import os, subprocess, time
import os.path as osp
from multiprocessing.dummy import Pool
root_path_home = '/mnt/HDD/edsr'
root_path_x4_video = '/mnt/HDD/edsr_4k/datasets/videos/test'
root_path_x4_png = '/mnt/HDD/edsr/datasets/Tencent/test'
root_path_x4_sr_png_experiment = '/mnt/HDD/edsr/experiment/rcantencent_x4_tmp'
root_path_x4_sr_video = '/mnt/HDD/edsr/datasets/Tencent/SR_4K'
srpngpathlist = []
gpunumber = 8

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def mkdir_plist(plist):
    for p in plist:
        os.makedirs(p,exist_ok=True)

def test(id,video):
    print(id)
    gpuid = id % gpunumber
    path_home = root_path_home
    path_x4_video = root_path_x4_video
    path_x4_png = osp.join(root_path_x4_png,video.split('.')[0])
    path_x4_sr_png = f"{root_path_x4_sr_png_experiment}_{video.split('.')[0]}/results-Demo"
    path_x4_sr_video = root_path_x4_sr_video
    mkdir_plist((path_home,path_x4_video,path_x4_png,path_x4_sr_png,path_x4_sr_video))
    # 准备工作，清理所有临时文件
    # del_file(path_x4_png)
    del_file(path_x4_sr_png)

    #  如果已完成测试，直接跳过
    if os.path.exists(os.path.join(path_x4_sr_video, video)):
        return

    # 1. 产生需要预测的png文件
    # cmd_png = 'ffmpeg -i ' + osp.join(path_x4_video, video) + ' -vsync 0 ' + osp.join(path_x4_png,'target%4d.png') + ' -y'
    # os.system(cmd_png)


    # 2. 调用算法产生超分后的png文件
    # pythonmain.py--templateRCAN--scale4--res_scale0.1--batch_size60--savercantencent_x4--chop--save_model--reset--n_threads16
    # cmd_edsr = f'python {path_home/src/}main.py --model EDSR --scale 4 --n_resblocks 16 --n_feats 128 --res_scale 0.1 --test_only --save_results --save edsrtencent_x4 --data_test Demo --pre_train ../experiment1012/edsrtencent_x4/model/model_best.pt --chop --dir_demo '+ path_x4_png


    cmd_rcan = f"CUDA_VISIBLE_DEVICES={gpuid} python {path_home}/src/main.py --template RCAN --scale 4 --res_scale 0.1  --test_only --save_results --save rcantencent_x4_tmp_{video.split('.')[0]} --data_test Demo --pre_train ../model_best.pt --chop --self_ensemble --dir_demo  "+ path_x4_png
    process_edsr = subprocess.Popen(cmd_rcan, shell=True)
    process_edsr.wait()

   
    save_path = path_x4_sr_video + '/' + video

    srpngpathlist.append((path_x4_sr_png,save_path))
    return 0

def tovideo(path):
    # 3. 编码超分后png文件为视频文件
    path_x4_sr_png = path[0]
    save_path = path[1]
    if os.path.exists(save_path):
        return
    cmd_encoder = 'ffmpeg -r 24000/1001 -i ' + path_x4_sr_png + f'/%4d_x4_SR.png  -vcodec libx265 -pix_fmt yuv422p -crf 5 -vf nlmeans -preset slower ' + save_path
    print(cmd_encoder)
    process_encoder = subprocess.Popen(cmd_encoder, shell=True)
    process_encoder.wait()
    if osp.exists(save_path) and (osp.getsize(save_path) > 61_000_000):
        os.remove(save_path)
        cmd_encoder = 'ffmpeg -r 24000/1001 -i ' + path_x4_sr_png + f'/%4d_x4_SR.png  -vcodec libx265 -pix_fmt yuv422p -crf 8 -vf nlmeans -preset slower ' + save_path
        process_encoder = subprocess.Popen(cmd_encoder, shell=True)
        process_encoder.wait()
    if osp.exists(save_path) and  (osp.getsize(save_path) > 61_000_000):
        os.remove(save_path)
        cmd_encoder = 'ffmpeg -r 24000/1001 -i ' + path_x4_sr_png + f'/%4d_x4_SR.png  -vcodec libx265 -pix_fmt yuv422p -crf 10 -vf nlmeans -preset slower ' + save_path
        process_encoder = subprocess.Popen(cmd_encoder, shell=True)
        process_encoder.wait()
    if osp.exists(save_path) and  (osp.getsize(save_path) > 61_000_000):
        os.remove(save_path)
        cmd_encoder = 'ffmpeg -r 24000/1001 -i ' + path_x4_sr_png + f'/%4d_x4_SR.png  -vcodec libx265 -pix_fmt yuv422p -crf 15 -vf nlmeans -preset slower ' + save_path
        process_encoder = subprocess.Popen(cmd_encoder, shell=True)
        process_encoder.wait()



if __name__ == "__main__":
    videolist = os.listdir(root_path_x4_video)
    for v in videolist:
        vname = v.split('.')[0]
        if not osp.exists(osp.join(root_path_x4_png,vname)):
            mkdir_plist((osp.join(root_path_x4_png,vname),))
            cmd_png = 'ffmpeg -i ' + root_path_x4_video + '/' + v + ' -vsync 0 ' + osp.join(root_path_x4_png,vname) + '/%4d.png -y'
            print(cmd_png)
            process_png = subprocess.Popen(cmd_png, shell=True)
            process_png.wait()
    print(len(videolist))

    pool = Pool(gpunumber)
    pool.map(lambda i: test(i,videolist[i]), range(len(videolist)),chunksize=1)
    pool.close()
    pool.join()

    pool = Pool()
    pool.map(tovideo,srpngpathlist)
    pool.close()
    pool.join()


