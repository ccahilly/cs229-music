import os
import subprocess

def convert_audio_files(audio_dir, converted_audio_dir, target_sample_rate=44100, target_channels=2):
    """
    Converts all audio files in a directory to a specified sample rate and channel configuration.
    
    Args:
        audio_dir (str): Path to the input directory containing audio files.
        converted_audio_dir (str): Path to the output directory to save converted files.
        target_sample_rate (int): Desired sample rate (default is 44100 Hz).
        target_channels (int): Desired number of audio channels (default is 2).
    """
    if not os.path.exists(converted_audio_dir):
        os.makedirs(converted_audio_dir)
    
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".wav"):  # Process only .wav files
            input_file_path = os.path.join(audio_dir, file_name)
            output_file_path = os.path.join(converted_audio_dir, file_name)
            
            # FFmpeg command for conversion
            command = [
                "ffmpeg",
                "-i", input_file_path,
                "-ar", str(target_sample_rate),
                "-ac", str(target_channels),
                output_file_path
            ]
            
            try:
                subprocess.run(command, check=True)
                print(f"Converted: {file_name} -> {output_file_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {file_name}: {e}")
    print("Audio conversion completed.")

# Define the directories
audio_dir = "../data/dpo-gen-output/"
converted_audio_dir = "../data/dpo-gen-output-converted/"

# Run the conversion
convert_audio_files(audio_dir, converted_audio_dir)