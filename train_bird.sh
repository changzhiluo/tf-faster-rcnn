cur_time=$(date "+%Y%m%d-%H%M%S")
src_dir="./output/res101"
des_dir="./output/res101_${cur_time}"
mv ${src_dir} ${des_dir}

rm -r ./data/cache

./experiments/scripts/train_faster_rcnn.sh 0 coco res101
