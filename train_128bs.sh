# Run generate_images.py using Docker
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    -v /media/8tsp/dataset:/datasets \
    -e HYDRA_FULL_ERROR=1 \
    -e OPENAI_LOGDIR="./.experiments/libritts-v5-5s-128bs" \
    -e OPENAI_LOG_FORMAT="stdout,csv,tensorboard" \
    --shm-size 128G \
    diffwave:latest \
    python ./audio_train.py --config-name="v5-5s-8bs"