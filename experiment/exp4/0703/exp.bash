for dir in /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/train/output/0703/*/; do
    dir_name=$(basename "$dir")
    python exp4.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/train/output/0703/"$dir_name" --device 6 --speed 1 0.1 1 --rate 1 0.3 1 --savepath 0703/"$dir_name"
    python exp4.py --target /mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/train/output/0703/"$dir_name" --device 6 --speed 1 20 1 --rate 1 3 1 --savepath 0703/"$dir_name"
done