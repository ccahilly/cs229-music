import os
from yt_dlp import YoutubeDL
from pydub import AudioSegment

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
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={ytid}'])
        return f"{ytid}.wav"  # Return the WAV filename directly
    except Exception as e:
        print(f"Error downloading audio for {ytid}: {e}")
        return None

def extract_audio_segment(input_file, start_s, end_s):
    try:
        # Load the webm file and extract the segment
        audio = AudioSegment.from_file(input_file, format="webm")  # Specify format as webm
        segment = audio[start_s * 1000:end_s * 1000]  # Convert seconds to milliseconds
        return segment
    except Exception as e:
        print(f"Error decoding audio file {input_file}: {e}")
        return None

def save_audio_segment(segment, ytid, start_s, end_s):
    if segment is not None:
        output_file = f"{ytid}_{start_s}_{end_s}.wav"  # Define output WAV filename
        segment.export(output_file, format="wav")  # Export as WAV
        return output_file
    return None

def main(ytid, start_s, end_s):
    audio_file = download_youtube_audio(ytid)
    if audio_file is None:
        return
    audio_segment = extract_audio_segment(audio_file, start_s, end_s)
    output_file = save_audio_segment(audio_segment, ytid, start_s, end_s)
    os.remove(audio_file)  # Remove the original webm file after processing
    return output_file

# Example usage with your dataset
ytid = "-0Gj8-vB1q4"  # Change this to another video ID if necessary
start_s = 30
end_s = 40

output_wav = main(ytid, start_s, end_s)
if output_wav:
    print(f"Extracted WAV file saved as: {output_wav}")
