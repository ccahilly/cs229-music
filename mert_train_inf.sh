#!/bin/bash

# Run mert_to_t5_train.py with frozen=True
echo "Running mert_to_t5_train.py with frozen=True..."
cd ~/cs229-music/mert_with_processor
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
python mert_to_t5_train.py --epochs=15 --last_epoch=0

# Check if the second script ran successfully
if [ $? -eq 0 ]; then
    echo "mert_to_t5_train.py (frozen=False) ran successfully."
else
    echo "Error in running mert_to_t5_train.py with frozen=False. Exiting script."
    exit 1
fi

# Run mert_to_t5_inference.py with frozen=True
echo "Running mert_to_t5_inference.py with frozen=True..."
python mert_to_t5_inference.py --frozen=frozen

# Check if the third script ran successfully
if [ $? -eq 0 ]; then
    echo "mert_to_t5_inference.py (frozen=True) ran successfully."
else
    echo "Error in running mert_to_t5_inference.py with frozen=True. Exiting script."
    exit 1
fi

# Run mert_to_t5_inference.py with frozen=False
echo "Running mert_to_t5_inference.py with frozen=False..."
python mert_to_t5_inference.py --frozen=unfrozen

# Check if the fourth script ran successfully
if [ $? -eq 0 ]; then
    echo "mert_to_t5_inference.py (frozen=False) ran successfully."
else
    echo "Error in running mert_to_t5_inference.py with frozen=False. Exiting script."
    exit 1
fi

echo "All training scripts completed successfully!"