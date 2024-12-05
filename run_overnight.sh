#!/bin/bash

# Run mert_to_t5_train.py with frozen=True
echo "Running mert_to_t5_train.py with frozen=True..."
cd ~/cs229-music/mert
python mert_to_t5_train.py --last_epoch 2 --epoch 13 --frozen True

# Check if the first script ran successfully
if [ $? -eq 0 ]; then
    echo "mert_to_t5_train.py (frozen=True) ran successfully."
else
    echo "Error in running mert_to_t5_train.py with frozen=True. Exiting script."
    exit 1
fi

# Run mert_to_t5_train.py with frozen=False
echo "Running mert_to_t5_train.py with frozen=False..."
python mert_to_t5_train.py --epochs=13 --last_epoch=2

# Check if the second script ran successfully
if [ $? -eq 0 ]; then
    echo "mert_to_t5_train.py (frozen=False) ran successfully."
else
    echo "Error in running mert_to_t5_train.py with frozen=False. Exiting script."
    exit 1
fi

# Run wav2vec_to_t5_train.py with frozen=True
echo "Running wav2vec_to_t5_train.py with frozen=True..."
cd ~/cs229-music/wav2vec-to-t5
python wav2vec_to_t5_train.py --epochs=13 --last_epoch=2 --frozen=True

# Check if the third script ran successfully
if [ $? -eq 0 ]; then
    echo "wav2vec_to_t5_train.py (frozen=True) ran successfully."
else
    echo "Error in running wav2vec_to_t5_train.py with frozen=True. Exiting script."
    exit 1
fi

# Run wav2vec_to_t5_train.py with frozen=False
echo "Running wav2vec_to_t5_train.py with frozen=False..."
python wav2vec_to_t5_train.py --epochs=13 --last_epoch=2

# Check if the fourth script ran successfully
if [ $? -eq 0 ]; then
    echo "wav2vec_to_t5_train.py (frozen=False) ran successfully."
else
    echo "Error in running wav2vec_to_t5_train.py with frozen=False. Exiting script."
    exit 1
fi

echo "All training scripts completed successfully!"