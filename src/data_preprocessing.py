from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import yaml
from loguru import logger
from rich.logging import RichHandler

logger.remove()
logger.add(RichHandler(), level="INFO")
logger.add("logs/prep_data.log", rotation="5 MB", level="INFO")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

CLIP_LENGTH_MS = 10 * 1000  
OVERLAP = 0.5  
INPUT_DIR = "data/splits"
OUTPUT_DIR = "data/prepared"
SILENCE_THRESHOLD = -50.0


def is_silent(audio, threshold=SILENCE_THRESHOLD):
    """Check if audio is mostly silent"""
    return audio.dBFS < threshold


def clip_audio_folder(folder_path, output_dir, clip_length_ms=CLIP_LENGTH_MS):
    files = list(Path(folder_path).glob("*.mp3"))
    label = folder_path.name
    split = folder_path.parent.name
    
    clips_created = 0
    songs_processed = 0
    songs_skipped = 0
    
    for file_path in tqdm(files, desc=f"Processing {split}/{label}"):
        try:
            song = AudioSegment.from_mp3(file_path)
            
            if len(song) < clip_length_ms:
                logger.warning(f"Skipping {file_path.name}: too short ({len(song)}ms)")
                songs_skipped += 1
                continue
            
            if is_silent(song):
                logger.warning(f"Skipping {file_path.name}: mostly silent")
                songs_skipped += 1
                continue
            
            song = song.set_channels(1)
            
            song_length_ms = len(song)
            base_name = file_path.stem
            output_folder = Path(output_dir) / split / label / base_name
            output_folder.mkdir(parents=True, exist_ok=True)
            
            clip_count = 0
            step_size = int(clip_length_ms * (1 - OVERLAP))
            
            for i in range(0, song_length_ms, step_size):
                end_time = min(i + clip_length_ms, song_length_ms)
                clip = song[i:end_time]
                
                if len(clip) < clip_length_ms:
                    silence_needed = clip_length_ms - len(clip)
                    silence = AudioSegment.silent(duration=silence_needed)
                    clip = clip + silence
                
                if is_silent(clip):
                    continue
                
                output_path = output_folder / f"{base_name}_{i//1000:04d}.wav"
                clip.export(output_path, format="wav", bitrate="192k")
                clip_count += 1
            
            clips_created += clip_count
            songs_processed += 1
            logger.debug(f"Created {clip_count} clips from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            songs_skipped += 1
    
    logger.info(f"{split}/{label}: Created {clips_created} clips from {songs_processed} songs ({songs_skipped} skipped)")
    return clips_created


if __name__ == "__main__":
    total_clips = 0
    
    for split in ["train", "val", "test"]:
        for label in ["positive", "negative"]:
            folder_path = Path(INPUT_DIR) / split / label
            if folder_path.exists():
                logger.info(f"Processing {split}/{label}...")
                clips = clip_audio_folder(folder_path, OUTPUT_DIR)
                total_clips += clips
            else:
                logger.warning(f"Folder {folder_path} does not exist. Skipping.")
    
    logger.info(f"Preprocessing completed! Total clips generated: {total_clips}")
