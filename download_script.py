import os
import json
import yt_dlp
import ffmpeg
import time
import random
from tqdm import tqdm

# Load MS-ASL dataset JSON
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Download video using yt-dlp
def download_video(url, output_path):
    try:
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_path,
            'quiet': False,
            'cookiefile': 'cookies.txt'  # ← now using the manually exported file
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"Error downloading video {url}: {e}")
        return False

# Clip the video based on start and end times using ffmpeg
def clip_video(input_path, output_path, start_time, end_time):
    try:
        ffmpeg.input(input_path, ss=start_time, to=end_time).output(output_path).run(overwrite_output=True)
        return True
    except ffmpeg.Error as e:
        print(f"Error clipping video: {e}")
        return False

# Throttle download speed by adding random delay between downloads
def throttle_download(min_delay=2, max_delay=6):
    delay = random.uniform(min_delay, max_delay)
    print(f"Sleeping for {delay:.2f} seconds to throttle...")
    time.sleep(delay)

# Main function to download and process clips
def download_and_process_clips(json_file, download_dir, max_videos=1000):
    start_index = 0
    data = load_json(json_file)[start_index:]

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    processed_count = 0
    total_samples = len(data)

    for offset, sample in enumerate(data):
        global_idx = offset + start_index
        if processed_count >= max_videos:
            break

        label = sample['label']
        if not (100 <= label < 150):
            continue  # Skip labels not between 50 and 99   

        print(f"\nProcessing video {processed_count + 1} of {max_videos} (Global idx: {global_idx})")

        video_url = sample['url']
        start_time = sample['start_time']
        end_time = sample['end_time']

        # New filenames based on global index and label
        video_filename = f"{global_idx}_{label}.mp4"
        output_video_path = os.path.join(download_dir, video_filename)

        # 1. Download the video
        if not download_video(video_url, output_video_path):
            print("Skipping to next video due to download failure.")
            continue

        # 2. Clip the video
        clip_filename = f"{global_idx}_{label}_clip.mp4"
        clip_path = os.path.join(download_dir, clip_filename)
        if not clip_video(output_video_path, clip_path, start_time, end_time):
            print("Skipping to next video due to clipping failure.")
            continue

        # 3. Delete the full video
        try:
            os.remove(output_video_path)
        except Exception as e:
            print(f"Warning: Could not delete {output_video_path}: {e}")

        processed_count += 1

        # 4. Random throttle
        throttle_download()

        # 5. Every 50 videos, take a longer rest (cooldown)
        if processed_count % 50 == 0:
            cooldown_time = random.uniform(20, 40)
            print(f"\nCooldown: Processed {processed_count} videos. Sleeping for {cooldown_time:.2f} seconds...\n")
            time.sleep(cooldown_time)

    print(f"\nFinished! Successfully processed {processed_count} clips.")

# Set paths
json_file_path = "C:/Users/emili/Downloads/MS-ASL/MS-ASL/MSASL_train.json"
download_dir = "50_to_100"

# Start downloading
download_and_process_clips(json_file_path, download_dir, max_videos=1000)
