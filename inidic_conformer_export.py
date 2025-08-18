import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0
EPS = 1e-10
N_FREQS = N_FFT // 2 + 1


def hz_to_mel_htk(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz_htk(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    n_freqs = n_fft // 2 + 1
    m = np.linspace(hz_to_mel_htk(fmin), hz_to_mel_htk(fmax), n_mels + 2)
    f = mel_to_hz_htk(m)
    bins = np.floor((n_fft + 1) * f / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        l = max(l, 0)
        c = max(c, 1)
        r = min(r, n_freqs - 1)
        if l == c:
            c = min(c + 1, n_freqs - 1)
        if c == r:
            r = min(r + 1, n_freqs - 1)
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


class ConvSTFTMel(nn.Module):
    def __init__(self):
        super().__init__()
        n = torch.arange(N_FFT, dtype=torch.float32)
        hann = 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (N_FFT - 1))
        self.register_buffer("hann", hann, persistent=False)

        k = torch.arange(N_FREQS, dtype=torch.float32).unsqueeze(-1)
        t = torch.arange(N_FFT, dtype=torch.float32).unsqueeze(0)
        ang = 2.0 * math.pi * k * t / N_FFT
        cos_k = torch.cos(ang) * hann
        sin_k = -torch.sin(ang) * hann

        self.register_buffer("kernel_cos", cos_k.unsqueeze(1), False)
        self.register_buffer("kernel_sin", sin_k.unsqueeze(1), False)

        mel = mel_filterbank(SR, N_FFT, N_MELS, F_MIN, F_MAX)
        self.register_buffer("mel_fb", torch.from_numpy(mel), False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(1)
        real = F.conv1d(x, self.kernel_cos, stride=HOP, padding=0)
        imag = F.conv1d(x, self.kernel_sin, stride=HOP, padding=0)
        power = real * real + imag * imag
        mel = torch.einsum("mf,bft->bmt", self.mel_fb, power)
        mel = torch.clamp(mel, min=EPS).log().transpose(1, 2)
        return mel


if __name__ == "__main__":
    model = ConvSTFTMel().eval()
    dummy = torch.randn(1, SR * 5, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        "preproc_fbank.onnx",
        input_names=["pcm"],
        output_names=["features"],
        dynamic_axes={"pcm": {1: "T"}, "features": {1: "n_frames"}},
        opset_version=17,
        do_constant_folding=True,
    )
