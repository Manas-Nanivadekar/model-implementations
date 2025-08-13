from pydub import AudioSegment

spacermilli = 1000
spacer = AudioSegment.silent(duration=spacermilli)

audio = AudioSegment.from_wav("audio.wav")
sound1 = audio[(4 * 60 + 6) * 1000 + 53 :]
sound1.export("sound1.wav", format="wav")

sound2 = audio[(3 * 60 + 42) * 1000 + 65 : (3 * 60 + 50) * 1000 + 750]
sound2 = spacer.append(sound2, crossfade=0)
sound2.export("sound2.wav", format="wav")

combined = sound1.overlay(sound2)
combined.export("combined.wav", format="wav")
