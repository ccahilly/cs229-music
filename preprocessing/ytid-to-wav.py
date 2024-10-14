import pandas as pd
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import os

# Run from preprocessing folder
# Note the resampling to 16 khz

def download_youtube_audio(ytid):
    try:
        # Set options to download as best audio quality and convert to WAV
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f"{ytid}.%(ext)s",
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'noplaylist': True,  # Don't download playlists
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={ytid}'])
        return f"{ytid}.wav"  # Return the WAV filename directly
    except Exception as e:
        print(f"Error downloading audio for {ytid}: {e}")
        return None

def extract_audio_segment(input_file, start_s, end_s, target_sample_rate=16000):
    try:
        # Load the webm file and extract the segment
        audio = AudioSegment.from_file(input_file, format="webm")  # Specify format as webm
        audio = audio.set_frame_rate(target_sample_rate)  # Resample to 16 kHz
        segment = audio[start_s * 1000:end_s * 1000]  # Convert seconds to milliseconds
        return segment
    except Exception as e:
        print(f"Error decoding audio file {input_file}: {e}")
        return None

def save_audio_segment(segment, ytid, start_s, end_s, output_dir):
    if segment is not None:
        output_file = f"{ytid}.wav"  # Define output WAV filename
        segment.export(output_dir + output_file, format="wav")  # Export as WAV
        return output_file
    return None

def single_download(ytid, start_s, end_s, output_dir):
    audio_file = download_youtube_audio(ytid)
    if audio_file is None:
        return
    audio_segment = extract_audio_segment(audio_file, start_s, end_s)
    output_file = save_audio_segment(audio_segment, ytid, start_s, end_s, output_dir)
    os.remove(audio_file)  # Remove the original webm file after processing
    return output_file

def main():
    # Load the CSV file
    csv_file = "../data/musiccaps/musiccaps-train-data.csv"
    data = pd.read_csv(csv_file)

    # Initialize counters and lists for success and failure
    success_count = 0
    failure_count = 0
    already_exists_count = 0
    failed_ytids = []

    # Specify the output directory
    output_dir = "../data/musiccaps/wav/"
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    # Loop through the first 10 rows in the CSV and process each one
    for index, row in data.iterrows():
        # if index >= 11:  # Break after processing 10 examples
        #     break

        ytid = row['ytid']
        start_s = row['start_s']
        end_s = row['end_s']

        if not os.path.exists(output_dir + str(ytid) + ".wav"):
            output_wav = single_download(ytid, start_s, end_s, output_dir)

            if output_wav:
                print(f"Successfully created WAV file: {output_wav}")
                success_count += 1
            else:
                print(f"Failed to create WAV file for ytid: {ytid}")
                failure_count += 1
                failed_ytids.append(ytid)
        else:
            already_exists_count += 1

    # Report the results
    print(f"\nTotal successes: {success_count}")
    print(f"Total failures: {failure_count}")
    print(f"Already existed: {already_exists_count}")

    # Write failed ytids to a file
    if failed_ytids:
        failed_ids_file = "../data/musiccaps/failed_ytids.txt"
        with open(failed_ids_file, 'w') as f:
            for failed_ytid in failed_ytids:
                f.write(f"{failed_ytid}\n")
        print(f"Failed ytids written to {failed_ids_file}")

main()