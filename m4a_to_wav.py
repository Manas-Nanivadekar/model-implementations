from pydub import AudioSegment
import sys
import os


def convert_m4a_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="m4a")

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    print(f"Converted: {input_path} → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_m4a_to_wav.py input.m4a output.wav")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    convert_m4a_to_wav(input_file, output_file)
