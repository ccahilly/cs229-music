import os
import pandas as pd
import simpleaudio as sa
import json
from generate_pairs import NUM_PAIRS_PER_CAPTION
from IPython.display import Audio, display

# Configurations
audio_dir = "../data/dpo-gen-output/"
caption_file = "../data/musiccaps-train-data.csv"
output_file = "./data/dpo-gen-output/human_labels.json"

print("Note that this must be run locally for the sound to work.")

# Load captions
captions_df = pd.read_csv(caption_file)

# Initialize labels
labels = []

# Helper function to play an audio file
def play_audio(file_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing audio: {e}")

# for f in os.listdir(audio_dir):
#     # Extract ytid and check if it's valid
#     if not f.endswith(".wav"):
#         continue
#     ytid = "-".join(f.split("-")[:-2])

#     for pair_idx in range(NUM_PAIRS_PER_CAPTION):  # Define NUM_PAIRS_PER_CAPTION
#         file_0 = os.path.join(audio_dir, f"{ytid}-pair{pair_idx}-0-converted.wav")
#         file_1 = os.path.join(audio_dir, f"{ytid}-pair{pair_idx}-1.wav")
        
#         # Skip if files are missing
#         if not os.path.exists(file_0) or not os.path.exists(file_1):
#             print(f"Missing files for {ytid} pair {pair_idx}. Skipping.")
#             continue

#         # Display caption
#         caption = captions_df[captions_df["ytid"] == ytid]["caption"].values[0]
#         print(f"\nCaption: {caption}")
#         # print(f"Pair {pair_idx}:")
#         print(f"1. {file_0}")
#         print(f"2. {file_1}")

#         # Playback loop
#         while True:
#             print("Type '1' to play first audio, '2' to play second audio, or 'q' to move to rating.")
#             choice = input("Play option (1/2/q): ").strip()
#             if choice == "1":
#                 play_audio(file_0)
#             elif choice == "2":
#                 play_audio(file_1)
#             elif choice == "q":
#                 break
#             else:
#                 print("Invalid input. Please try again.")

#         # Collect human preference
#         while True:
#             preference = input("Enter your preference (0 for first, 1 for second, -1 for no preference): ").strip()
#             if preference in {"0", "1", "-1"}:
#                 labels.append({"ytid": ytid, "pair_idx": pair_idx, "preference": int(preference)})
#                 break
#             else:
#                 print("Invalid input. Please enter 0, 1, or -1.")

# Function to get the caption given the ytid
def get_caption(ytid):
    row = captions_df[captions_df['ytid'] == ytid]
    return row['caption'].iloc[0] if not row.empty else "Caption not found"

# Iterate over generated audio pairs
for ytid in os.listdir(audio_dir):
    # Extract ytid and check if it's valid
    if not ytid.endswith(".wav"):
        continue
    ytid_base = "-".join(ytid.split("-")[:-2])

    # Retrieve caption
    caption = get_caption(ytid_base)
    if caption == "Caption not found":
        continue

    # Loop through pairs for this ytid
    for pair_index in range(NUM_PAIRS_PER_CAPTION):
        # Audio file paths
        audio_file_0 = os.path.join(audio_dir, f"{ytid_base}-pair{pair_index}-0.wav")
        audio_file_1 = os.path.join(audio_dir, f"{ytid_base}-pair{pair_index}-1.wav")

        # Ensure both files exist
        if not (os.path.exists(audio_file_0) and os.path.exists(audio_file_1)):
            continue

        # Show caption
        print(f"\nCaption: {caption}\n")

        # Play audio files
        print(f"Playing pair {pair_index}:")
        print("Audio 1:")
        display(Audio(audio_file_0))
        print("Audio 2:")
        display(Audio(audio_file_1))

        # Collect user input
        while True:
            user_input = input("Type 0 if you prefer Audio 1, 1 for Audio 2, or -1 for no preference: ").strip()
            if user_input in {"0", "1", "-1"}:
                break
            print("Invalid input. Please enter 0, 1, or -1.")

        # Store the result
        labels.append({
            "ytid": ytid_base,
            "pair_index": pair_index,
            "audio_file_0": audio_file_0,
            "audio_file_1": audio_file_1,
            "caption": caption,
            "preference": int(user_input)
        })

# Save labeled data
with open(output_file, "w") as f:
    # json.dump(labels, f, indent=2)
    json.dump(labels, f, indent=4)

print(f"Human labels saved to {output_file}")
