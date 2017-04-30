

# make pre plocess data

    python make_train_data.py 300image
    python compute_mean.py train.txt

# train

    python train_imagenet.py ./train.txt ./test.txt -m ./mean.npy -g 0 -E 400 -a alex

# test

    python test_imagenet.py --test image_list.txt -g 0 -E 1 -m mean.npy --initmodel alex_400e_png_model.h5 -a alex
