# cs229-music

# DATA FOLDER
    # MUSICCAPS FOLDER
        # data/musiccaps/musiccaps-train-data.csv
        Downloaded from Kaggle (https://www.kaggle.com/datasets/googleai/musiccaps)
        Contains YouTube ID, start_s, end_s, audioset_positive_labels, aspect_list, caption, 
        author_id, is_balanced_subset, is_audioset_eval

        # WAV FOLDER
        Contains all of the wav files for musiccaps-train-data.csv
        Generate by running ytid-to-wav.py from the preprocessing folder

# PREPROCESSING FOLDER
    # ytid-to-wav.py
    Generates the wav files for ../data/musiccaps/musiccaps-train-data.csv