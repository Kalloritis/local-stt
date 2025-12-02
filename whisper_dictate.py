import argparse
import sys
import time
import queue

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


def notify_start():
    """Make a short beep when recording really starts."""
    try:
        import platform
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 150)  # 1000 Hz, 150 ms
        else:
            # Fallback: console bell (no-op if console hidden)
            print("\a", file=sys.stderr, flush=True)
    except Exception:
        # Don't let a beep failure kill the script
        pass

def list_devices():
    print("Available input devices:\n")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"[{idx}] {dev['name']}")
    print("\nUse --device-index <index> to select one.")
    sys.exit(0)


def record_audio_vad(
    samplerate: int,
    device_index: int | None,
    silence_threshold: float,
    silence_seconds: float,
    max_seconds: float,
    min_seconds_before_silence_stop: float = 10.0,  # don't stop on silence before 10s total
):
    """
    Record audio until:
      - RMS amplitude stays below `silence_threshold`
        for at least `silence_seconds` *and* we've been
        recording at least `min_seconds_before_silence_stop`, OR
      - `max_seconds` is reached.
    """
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    if device_index is not None:
        sd.default.device = device_index

    q_audio = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        q_audio.put(indata.copy())

    print(
        f"Recording... (stop = {silence_seconds}s of silence or {max_seconds}s max)",
        file=sys.stderr,
    )

    audio_chunks = []
    start_time = time.time()
    silence_start = None

    notify_start()  # beep when recording actually starts

    with sd.InputStream(
        callback=callback,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device_index,
    ):
        while True:
            try:
                indata = q_audio.get(timeout=1.0)
            except queue.Empty:
                indata = np.zeros((int(samplerate * 0.1), 1), dtype="float32")

            audio_chunks.append(indata)
            now = time.time()
            elapsed = now - start_time

            rms = float(np.sqrt(np.mean(indata**2))) if indata.size > 0 else 0.0

            # Only allow silence-based stop after we've been going for a while
            if elapsed >= min_seconds_before_silence_stop:
                if rms < silence_threshold:
                    if silence_start is None:
                        silence_start = now
                    elif now - silence_start >= silence_seconds:
                        print("Detected sustained silence. Stopping.", file=sys.stderr)
                        break
                else:
                    silence_start = None

            if elapsed >= max_seconds:
                print("Reached max recording time. Stopping.", file=sys.stderr)
                break

    if not audio_chunks:
        return np.zeros((0,), dtype="float32")

    audio = np.concatenate(audio_chunks, axis=0).astype("float32").squeeze()
    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Local Whisper dictation: record audio until silence and print text."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input devices and exit.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Input device index (see --list-devices).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        help="Whisper model size: tiny, base, small, medium, large-v2, etc.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (default: en).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Compute device: "auto", "cpu", or "cuda".',
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.01,
        help="RMS threshold below which audio is considered silence (default: 0.01).",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=1.5,
        help="Seconds of continuous silence before stopping (default: 1.5).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=300.0,
        help="Maximum recording length in seconds as safety cap (default: 300).",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()

    # Load model
    print(
        f"Loading faster-whisper model '{args.model_size}' on {args.device}...",
        file=sys.stderr,
    )

    # Let faster-whisper handle "auto" itself; don't pass None.
    device_arg = args.device  # "auto", "cpu", or "cuda"

    compute_type = (
        "int8_float16" if device_arg == "cuda" else "int8"
    )

    model = WhisperModel(
        args.model_size,
        device=device_arg,  # <- no more None
        compute_type=compute_type,
    )

    audio = record_audio_vad(
        samplerate=16000,
        device_index=args.device_index,
        silence_threshold=args.silence_threshold,
        silence_seconds=args.silence_seconds,
        max_seconds=args.max_seconds,
        # Optional override if you want:
        # min_seconds_before_silence_stop=10.0,
    )

    if audio.size == 0:
        print("", end="")
        return

    segments, info = model.transcribe(
        audio,
        language=args.language,
        beam_size=5,
        vad_filter=True,
    )

    text = "".join(segment.text for segment in segments).strip()
    print(text)


if __name__ == "__main__":
    main()
