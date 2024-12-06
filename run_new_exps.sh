#!/bin/bash

# Run mert_to_t5_train.py with frozen=False
echo "Running mert_to_t5_train.py with frozen=False..."
cd ~/cs229-music/mert_with_processor
python mert_to_t5_train.py --last_epoch 4 --epoch 11

# Check if the first script ran successfully
if [ $? -eq 0 ]; then
    echo "mert_to_t5_train.py (frozen=True) ran successfully."
else
    echo "Error in running mert_to_t5_train.py with frozen=True. Exiting script."
    exit 1
fi

# Run wav2vec_to_t5_train.py with all_frozen=True
echo "Running wav2vec_to_t5_train.py with all frozen true..."
cd ~/cs229-music/wav2vec-to-t5
python wav2vec_to_t5_train.py --freeze_embed True --freeze_t5 True --epochs=14 --last_epoch=1

# Check if the second script ran successfully
if [ $? -eq 0 ]; then
    echo "wav2vec_to_t5_train.py (all frozen true) ran successfully."
else
    echo "Error in running av2vec_to_t5_train.py (all frozen true). Exiting script."
    exit 1
fi

# Run mert_to_t5_train.py with all_frozen=True
echo "Running mert_to_t5_train.py with all frozen true..."
cd ~/cs229-music/mert_with_processor
python mert_to_t5_train.py --epochs=15 --freeze_embed=True --freeze_t5=True

# Check if the third script ran successfully
if [ $? -eq 0 ]; then
    echo "mert_to_t5_train.py (frozen=True) ran successfully."
else
    echo "Error in running mert_to_t5_train.py with frozen=True. Exiting script."
    exit 1
fi