# for train
# python train_caption.py -c config/caption/paper2020/yc2_100m_coot_vidclip_mart.yaml
python train_caption.py -c config/caption/paper2020/yc2_100m_coot_clip_mart.yaml -lw 0.01
#for test
python show_caption.py -m base