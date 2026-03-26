import argparse
import os
import shutil
import srt
import requests
import dashscope
import concurrent.futures

from dataclasses import dataclass
from tqdm import tqdm
from datetime import timedelta
from collections import Counter
from typing import List, Optional
from urllib.parse import urlparse
from silero_vad import load_silero_vad
from qwen3_asr_toolkit.qwen3asr import QwenASR, QwenASRAligner
from qwen3_asr_toolkit.audio_tools import load_audio, process_vad, save_audio_file, WAV_SAMPLE_RATE


@dataclass
class SegmentResult:
    index: int
    start_time: float
    end_time: float
    language: str
    text: str


@dataclass
class TranscriptionResult:
    input_file: str
    language: str
    text: str
    segments: List[SegmentResult]
    text_output_path: Optional[str] = None
    srt_output_path: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Python toolkit for the Qwen3-ASR API—parallel high‑throughput calls, robust long‑audio transcription, multi‑sample‑rate support."
    )
    parser.add_argument("--input-file", '-i', type=str, required=True, help="Input media file path")
    parser.add_argument("--context", '-c', type=str, default="", help="Any text context content for Qwen3-ASR-Flash")
    parser.add_argument("--dashscope-api-key", '-key', type=str, help="DashScope API key")
    parser.add_argument("--num-threads", '-j', type=int, default=4, help="Number of threads to use for parallel calls")
    parser.add_argument("--vad-segment-threshold", '-d', type=int, default=120, help="Segment threshold seconds for VAD")
    parser.add_argument("--tmp-dir", '-t', type=str, default=os.path.join(os.path.expanduser("~"), "qwen3-asr-cache"), help="Temp directory path")
    parser.add_argument("--save-srt", '-srt', action="store_true", help="Save SRT subtitle file")
    parser.add_argument("--silence", '-s', action="store_true", help="Reduce the output info on the terminal")
    return parser.parse_args()

def parse_subtitles(aligner, subtitles, start_time, end_time, content, wav_path, language):
    result = aligner.align(
        wav_path,
        content,
        lang=language,
    )
    
    """
    Format of Output:
    item: text, start, end
    """
    word_alignments = result[0].items
    max_chars = 100
    
    current_chars = 0
    srt_sections = []
    current_section_words = []

    for item in word_alignments:
        word_text = item.text
        if current_chars + len(word_text) + 1 > max_chars and current_section_words:
            # Store the completed section
            srt_sections.append(current_section_words)
            current_section_words = [item]
            current_chars = len(word_text)
        else:
            current_section_words.append(item)
            current_chars += len(word_text) + 1

    if current_section_words:
        srt_sections.append(current_section_words)

    # Fall back to segment-level subtitles if forced alignment returns no words.
    if not srt_sections:
        subtitles.append(srt.Subtitle(
            index=len(subtitles) + 1,
            start=timedelta(seconds=start_time),
            end=timedelta(seconds=end_time),
            content=content
        ))
        return

    # Write subtitles from @srt_sections to the final srt file
    for section in srt_sections:
        section_start_time = start_time + section[0].start_time
        section_end_time = start_time + section[-1].end_time
        section_content = " ".join([word.text for word in section])
        subtitles.append(srt.Subtitle(
            index=len(subtitles) + 1,
            start=timedelta(seconds=section_start_time),
            end=timedelta(seconds=section_end_time),
            content=section_content
        ))
    
    

def _validate_input_exists(input_file: str):
    if input_file.startswith(("http://", "https://")):
        try:
            response = requests.head(input_file, allow_redirects=True, timeout=5)
            if response.status_code >= 400:
                raise FileNotFoundError(f"returned status code {response.status_code}")
        except Exception as e:
            raise FileNotFoundError(f"HTTP link {input_file} does not exist or is inaccessible: {str(e)}")
    elif not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file \"{input_file}\" does not exist!")


def _default_text_output_path(input_file: str) -> str:
    if os.path.exists(input_file):
        return os.path.splitext(input_file)[0] + ".txt"

    url_path = os.path.splitext(urlparse(input_file).path)[0].split('/')[-1]
    base_name = url_path if url_path else "transcription"
    return base_name + ".txt"


def transcribe(
    input_file: str,
    context: str = "",
    dashscope_api_key: Optional[str] = None,
    num_threads: int = 4,
    vad_segment_threshold: int = 120,
    tmp_dir: str = os.path.join(os.path.expanduser("~"), "qwen3-asr-cache"),
    save_srt: bool = False,
    silence: bool = False,
    model: str = "qwen3-asr-flash",
    save_text: bool = True,
    cleanup_tmp: bool = True,
) -> TranscriptionResult:
    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")
    if vad_segment_threshold <= 0:
        raise ValueError("vad_segment_threshold must be > 0")

    _validate_input_exists(input_file)

    if dashscope_api_key:
        dashscope.api_key = dashscope_api_key
    else:
        assert "DASHSCOPE_API_KEY" in os.environ, f"Please set DASHSCOPE_API_KEY as an environment variable, or specify it with '-key' argument"

    qwen3asr = QwenASR(model=model)

    wav = load_audio(input_file)
    if not silence:
        print(f"Loaded wav duration: {len(wav) / WAV_SAMPLE_RATE:.2f}s")

    # Segment wav exceeding 3 minutes
    if len(wav) / WAV_SAMPLE_RATE >= 180:
        if not silence:
            print(f"Wav duration is longer than 3 min, initializing Silero VAD model for segmenting...")
        worker_vad_model = load_silero_vad(onnx=True)
        wav_list = process_vad(wav, worker_vad_model, segment_threshold_s=vad_segment_threshold)
        if not silence:
            print(f"Segmenting done, total segments: {len(wav_list)}")
    else:
        wav_list = [(0, len(wav), wav)]

    # Save processed audio to tmp dir
    wav_name = os.path.basename(input_file)
    wav_dir_name = os.path.splitext(wav_name)[0]
    save_dir = os.path.join(tmp_dir, wav_dir_name)

    wav_path_list = []
    for idx, (_, _, wav_data) in enumerate(wav_list):
        wav_path = os.path.join(save_dir, f"{wav_name}_{idx}.wav")
        save_audio_file(wav_data, wav_path)
        wav_path_list.append(wav_path)

    # Multithread call qwen3-asr-flash api
    results = []
    languages = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_dict = {
            executor.submit(qwen3asr.asr, wav_path, context): idx
            for idx, wav_path in enumerate(wav_path_list)
        }
        pbar = None
        if not silence:
            pbar = tqdm(total=len(future_dict), desc="Calling Qwen3-ASR-Flash API")
        for future in concurrent.futures.as_completed(future_dict):
            idx = future_dict[future]
            language, recog_text = future.result()
            results.append((idx, language, recog_text))
            languages.append(language)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

    # Sort and splice in the original order
    results.sort(key=lambda x: x[0])
    full_text = " ".join(text for _, _, text in results)
    language = Counter(languages).most_common(1)[0][0]

    segment_results = []
    for idx, seg_language, seg_text in results:
        segment_results.append(SegmentResult(
            index=idx,
            start_time=wav_list[idx][0] / WAV_SAMPLE_RATE,
            end_time=wav_list[idx][1] / WAV_SAMPLE_RATE,
            language=seg_language,
            text=seg_text,
        ))

    if not silence:
        print(f"Detected Language: {language}")
        print(f"Full Transcription: {full_text}")

    text_output_path = None
    if save_text:
        text_output_path = _default_text_output_path(input_file)
        with open(text_output_path, 'w') as f:
            f.write(language + '\n')
            f.write(full_text + '\n')
        if not silence:
            print(f"Full transcription of \"{input_file}\" from Qwen3-ASR-Flash API saved to \"{text_output_path}\"!")

    srt_output_path = None
    if save_srt:
        if not save_text:
            text_output_path = _default_text_output_path(input_file)
        assert text_output_path is not None

        subtitles = []
        aligner = QwenASRAligner()

        for idx, result in enumerate(results):
            start_time = wav_list[idx][0] / WAV_SAMPLE_RATE
            end_time = wav_list[idx][1] / WAV_SAMPLE_RATE
            seg_language = result[1]
            content = result[2]
            
            parse_subtitles(
                aligner = aligner,
                subtitles = subtitles,
                start_time = start_time,
                end_time = end_time,
                content = content,
                wav_path = wav_path_list[idx],
                language = seg_language,
            )

        # Guarantee SRT indices are strictly sequential.
        for i, sub in enumerate(subtitles, start=1):
            sub.index = i

        final_srt_content = srt.compose(subtitles)
        srt_output_path = os.path.splitext(text_output_path)[0] + ".srt"
        with open(srt_output_path, 'w') as f:
            f.write(final_srt_content)
        if not silence:
            print(f"SRT subtitles of \"{input_file}\" from Qwen3-ASR-Flash API saved to \"{srt_output_path}\"!")

    if cleanup_tmp:
        shutil.rmtree(save_dir, ignore_errors=True)

    return TranscriptionResult(
        input_file=input_file,
        language=language,
        text=full_text,
        segments=segment_results,
        text_output_path=text_output_path,
        srt_output_path=srt_output_path,
    )


def main():
    args = parse_args()
    transcribe(
        input_file=args.input_file,
        context=args.context,
        dashscope_api_key=args.dashscope_api_key,
        num_threads=args.num_threads,
        vad_segment_threshold=args.vad_segment_threshold,
        tmp_dir=args.tmp_dir,
        save_srt=args.save_srt,
        silence=args.silence,
    )


if __name__ == '__main__':
    main()
