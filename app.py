import io
import wave
import numpy as np
import streamlit as st

# ============================
# 1) HEDEF DEĞERLER (NORMAL DURUM)
# ============================
TARGETS = {
    "anxious": 2.0,        # kaygı düşük
    "depressed": 2.0,      # depresyon düşük
    "frightened": 2.0,     # korku düşük
    "disorganized": 2.0,   # dağınıklık düşük (coherent)
    "hopeful": 7.0,        # umut yüksek
}

def clamp(x, mn=0.0, mx=10.0):
    return max(mn, min(mx, x))


# ============================
# 2) PARAMETRE TASARIMI
# ============================
def design_music_params(raw_mood: dict):
    a = clamp(raw_mood["anxious"])
    d = clamp(raw_mood["depressed"])
    f = clamp(raw_mood["frightened"])
    dis = clamp(raw_mood["disorganized"])
    h = clamp(raw_mood["hopeful"])

    # "Patoloji" (anxious + depressed + frightened)
    pathology = (a + d + f) / 3.0  # 0–10

    # Tempo (önce normal hesapla, sonra %33 yavaşlat)
    bpm_raw = 75 - 15 * (pathology / 10.0)
    bpm_raw = max(60, min(82, bpm_raw))
    bpm = bpm_raw * (2.0 / 3.0)              # ~%33 yavaş
    bpm = max(48, min(68, bpm))              # lo-fi yavaş aralık

    # Temel frekans
    base_freq = 220 + 50 * (h / 10.0) - 40 * (d / 10.0)
    base_freq = max(140, min(380, base_freq))

    # Majör / minör
    if h >= d:
        scale_type = "major"
    else:
        scale_type = "minor"

    # Disorganized: 0–1 arası kaos seviyesi
    dis_chaos = dis / 10.0
    # Vibrato üst sınırı
    max_vibrato = 0.02 + 0.05 * dis_chaos

    return {
        "bpm": bpm,
        "base_freq": base_freq,
        "scale_type": scale_type,
        "dis_chaos": dis_chaos,
        "max_vibrato": max_vibrato,
    }


# ============================
# 3) DİZİ & AKOR YAPISI
# ============================
def build_scale(base_freq: float, scale_type: str):
    if scale_type == "minor":
        intervals = np.array([1.0, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5, 2.0])
    else:
        intervals = np.array([1.0, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2.0])

    return base_freq * intervals


def build_chord_progression(scale_type: str, rng: np.random.Generator):
    if scale_type == "minor":
        patterns = [
            [0, 5, 6, 4],
            [0, 3, 4, 0],
            [0, 6, 5, 4],
            [0, 5, 3, 4],
            [0, 2, 5, 4],
            [0, 3, 6, 4],
            [0, 4, 5, 3],
            [0, 6, 4, 5],
        ]
    else:
        patterns = [
            [0, 4, 5, 3],
            [0, 5, 3, 4],
            [0, 3, 4, 0],
            [0, 4, 1, 5],
            [0, 5, 1, 4],
            [0, 2, 5, 4],
            [0, 3, 5, 4],
        ]

    pattern = patterns[rng.integers(0, len(patterns))]
    return pattern


# ============================
# 4) LEAD (ARP) VARYANT SEÇİMİ
# ============================
def choose_lead_clean_freq(chord_degrees, scale_freqs, rng: np.random.Generator):
    n = len(scale_freqs)
    candidates = []

    deg1 = chord_degrees[1] + 7
    candidates.append(scale_freqs[deg1 % n])

    deg2 = chord_degrees[2] + 7
    candidates.append(scale_freqs[deg2 % n])

    deg3 = chord_degrees[0] + 14
    candidates.append(scale_freqs[deg3 % n])

    deg4 = chord_degrees[0] + 9
    candidates.append(scale_freqs[deg4 % n])

    deg5 = chord_degrees[1] + 14
    candidates.append(scale_freqs[deg5 % n])

    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


# ============================
# 5) LO-FI SYNTH TONE SENTEZİ (DAHA DA TATLI)
# ============================
def synth_voice(inst_freq, t_step, harmonics=(1.0, 0.35, 0.12)):
    """
    Daha da tatlı, pürüzsüz lo-fi ton:
    - Üst harmonikler daha düşük
    - Sine ağırlıklı, kremamsı ton
    """
    inst_freq = np.asarray(inst_freq)
    if inst_freq.ndim == 0:
        inst_freq = inst_freq * np.ones_like(t_step)

    sig = np.zeros_like(t_step, dtype=np.float32)
    total_amp = sum(abs(a) for a in harmonics) or 1.0

    for k, amp in enumerate(harmonics, start=1):
        sig += amp * np.sin(2.0 * np.pi * (k * inst_freq) * t_step)

    sig /= total_amp
    return sig


# ============================
# 6) LO-FI DRUM & NOISE HELPER'LARI
# ============================
def make_kick(t_step, freq=60.0):
    env = np.exp(-np.linspace(0.0, 6.0, len(t_step)))
    sig = np.sin(2 * np.pi * freq * t_step) * env
    return sig

def make_snare(t_step, rng: np.random.Generator):
    env = np.exp(-np.linspace(0.0, 8.0, len(t_step)))
    noise = rng.normal(0.0, 1.0, len(t_step))
    noise = noise / (np.max(np.abs(noise)) + 1e-6)
    hi = np.sin(2 * np.pi * 160.0 * t_step)  # daha sıcak, daha az parlak
    sig = (0.8 * noise + 0.2 * hi) * env
    return sig

def lofi_filter(signal: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Daha güçlü low-pass / blur filtresi:
    - Tizleri iyice yumuşatır
    - Genel ses dokusunu daha pürüzsüz yapar
    """
    if kernel_size <= 1:
        return signal
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    out = np.convolve(signal, kernel, mode="same")
    return out


# ============================
# 7) SES ÜRETİMİ (LO-FI POP + 3 FAZ + ÇOK YUMUŞAK BİTİŞ)
# ============================
def generate_normalizing_music(raw_mood: dict, duration_s: float = 30.0, sr: int = 44100):
    rng = np.random.default_rng()

    params = design_music_params(raw_mood)

    bpm = params["bpm"]
    base_freq = params["base_freq"]
    scale_type = params["scale_type"]
    dis_chaos = params["dis_chaos"]
    max_vibrato = params["max_vibrato"]

    # --- LO-FI STİL SEÇİMİ (RANDOM) ---
    style_key = rng.choice(["lofi_soft", "lofi_dusty", "lofi_dark"])
    style_labels = {
        "lofi_soft": "Yumuşak lo-fi",
        "lofi_dusty": "Tozlu / vinil lo-fi",
        "lofi_dark": "Daha karanlık gece lo-fi",
    }
    style_desc = style_labels[style_key]

    if style_key == "lofi_dark":
        base_freq *= 0.9
        scale_type = "minor"
        max_vibrato *= 1.1
    elif style_key == "lofi_dusty":
        bpm = max(48, bpm * 0.95)
    elif style_key == "lofi_soft":
        bpm = max(48, min(66, bpm))

    scale_freqs = build_scale(base_freq, scale_type)
    chord_roots = build_chord_progression(scale_type, rng)

    sec_per_beat = 60.0 / bpm
    step_beats = 0.5
    step_sec = sec_per_beat * step_beats
    step_samples = int(sr * step_sec)
    n_steps = max(1, int(np.ceil(duration_s / step_sec)))

    segments = []

    melody_pattern = [0, 1, 2, 1, 0, 1, 3, 1]
    steps_per_bar = int(4.0 / step_beats)

    for i in range(n_steps):
        if n_steps > 1:
            prog = i / (n_steps - 1)
        else:
            prog = 0.0

        in_relax_zone = prog >= 0.40
        in_last_bar = (i >= n_steps - steps_per_bar)

        # 0–0.20: orijinal
        # 0.20–0.40: geçiş (1→0)
        # 0.40–1.00: düzelmiş (0)
        if prog < 0.20:
            phase_factor = 1.0
        elif prog < 0.40:
            inner = (prog - 0.20) / 0.20
            phase_factor = 1.0 - inner
        else:
            phase_factor = 0.0

        chaos = phase_factor * dis_chaos
        vibrato_depth = phase_factor * max_vibrato

        bar_idx = i // steps_per_bar
        step_in_bar = i % steps_per_bar

        if in_last_bar:
            chord_root_degree = chord_roots[0]
        else:
            chord_root_degree = chord_roots[bar_idx % len(chord_roots)]

        chord_degrees = [
            chord_root_degree % 7,
            (chord_root_degree + 2) % 7,
            (chord_root_degree + 4) % 7,
            (chord_root_degree) % 7 + 7
        ]

        pattern_slot = melody_pattern[i % len(melody_pattern)]
        if in_last_bar:
            pattern_slot = 0 if (i % 2 == 0) else 3

        clean_degree = chord_degrees[pattern_slot % len(chord_degrees)]
        clean_idx = clean_degree % len(scale_freqs)
        freq_clean = scale_freqs[clean_idx]

        if rng.random() < chaos:
            rand_idx = int(rng.integers(0, len(scale_freqs)))
            freq_melody = scale_freqs[rand_idx]
        else:
            freq_melody = freq_clean

        t_step = np.linspace(0, step_sec, step_samples, endpoint=False)

        if vibrato_depth > 0:
            vibrato_freq = 5.0
            vibrato = np.sin(2 * np.pi * vibrato_freq * t_step)
            inst_freq_mel = freq_melody * (1.0 + vibrato_depth * vibrato)
        else:
            inst_freq_mel = freq_melody

        if in_relax_zone:
            mel_harm = (1.0, 0.30, 0.10)
        else:
            mel_harm = (1.0, 0.35, 0.12)

        mel_sig = synth_voice(inst_freq_mel, t_step, harmonics=mel_harm)

        pad_freqs = [
            scale_freqs[chord_degrees[0] % len(scale_freqs)],
            scale_freqs[chord_degrees[1] % len(scale_freqs)],
            scale_freqs[chord_degrees[2] % len(scale_freqs)],
        ]

        if in_relax_zone:
            pad_harm = (1.0, 0.30, 0.10)
        else:
            pad_harm = (1.0, 0.35, 0.15)

        pad_sig = (
            synth_voice(pad_freqs[0], t_step, harmonics=pad_harm) +
            synth_voice(pad_freqs[1], t_step, harmonics=pad_harm) +
            synth_voice(pad_freqs[2], t_step, harmonics=pad_harm)
        ) / 3.0

        bass_freq = pad_freqs[0] / 2.0
        bass_sig = synth_voice(bass_freq, t_step, harmonics=(1.0, 0.25, 0.0))

        # Lead
        lead_clean_freq = choose_lead_clean_freq(chord_degrees, scale_freqs, rng)
        semitone_offset = int(rng.integers(-4, 5))
        lead_disson_freq = lead_clean_freq * (2.0 ** (semitone_offset / 12.0))
        lead_disson_prob = chaos

        if rng.random() < lead_disson_prob:
            freq_lead = lead_disson_freq
        else:
            freq_lead = lead_clean_freq

        if vibrato_depth > 0:
            vibrato2 = np.sin(2 * np.pi * 4.0 * t_step)
            inst_freq_lead = freq_lead * (1.0 + 0.5 * vibrato_depth * vibrato2)
        else:
            inst_freq_lead = freq_lead

        if in_relax_zone:
            lead_harm = (1.0, 0.35, 0.12)
        else:
            lead_harm = (1.0, 0.40, 0.15)

        lead_sig = synth_voice(inst_freq_lead, t_step, harmonics=lead_harm)

        # Lo-fi drum
        kick_sig = np.zeros_like(t_step)
        snare_sig = np.zeros_like(t_step)

        if step_in_bar in (0, 4):
            kick_sig = make_kick(t_step, freq=55.0)

        if step_in_bar in (2, 6):
            snare_sig = make_snare(t_step, rng)

        if step_samples > 4:
            base_env = np.hanning(step_samples)
            env_step = base_env
        else:
            env_step = np.ones_like(mel_sig)

        if in_relax_zone:
            drum_scale = 0.30
            lead_scale = 0.60
        else:
            drum_scale = 0.80
            lead_scale = 0.80

        step_signal = (
            0.44 * mel_sig +
            0.30 * pad_sig +
            0.24 * bass_sig +
            0.14 * lead_scale * lead_sig +
            0.08 * drum_scale * kick_sig +
            0.06 * drum_scale * snare_sig
        ) * env_step

        # Çok hafif saturation → pürüzsüzlük korunarak sıcaklık
        step_signal = np.tanh(step_signal * 1.02)

        segments.append(step_signal)

    signal = np.concatenate(segments)

    # Global fade in/out
    n_samples = len(signal)
    fade_len = min(int(sr * 1.0), n_samples // 4)
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)
    envelope = np.ones_like(signal)
    envelope[:fade_len] *= fade_in
    envelope[-fade_len:] *= fade_out
    signal = signal * envelope

    # Hafif vinil noise (daha da az)
    rng = np.random.default_rng()
    noise = rng.normal(0.0, 1.0, n_samples)
    noise = noise / (np.max(np.abs(noise)) + 1e-6)
    signal = signal + 0.010 * noise

    # Güçlü low-pass → daha pürüzsüz
    signal = lofi_filter(signal, kernel_size=15)

    # Reverb
    delay1 = int(sr * 0.09)
    delay2 = int(sr * 0.18)

    rev_len = n_samples + delay2
    wet = np.zeros(rev_len, dtype=np.float32)

    wet[:n_samples] += signal
    wet[delay1:delay1 + n_samples] += 0.24 * signal
    wet[delay2:delay2 + n_samples] += 0.16 * signal

    signal = wet
    n_samples = len(signal)

    fade_len2 = min(int(sr * 1.5), n_samples)
    fade_out2 = np.linspace(1.0, 0.0, fade_len2)
    signal[-fade_len2:] *= fade_out2

    max_val = np.max(np.abs(signal)) + 1e-6
    signal = 0.9 * signal / max_val

    audio_int16 = (signal * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    audio_bytes = buf.read()

    info = {
        "style_key": style_key,
        "style_desc": style_desc,
    }
    return audio_bytes, info


# ============================
# 8) STREAMLIT ARAYÜZÜ
# ============================
st.set_page_config(page_title="Lo-fi Coherence Mood Normalizer", page_icon="🎵", layout="centered")
st.title("🎵 Şifalı Müzik Uygulaması")

st.write(
    """
Bu uygulama, girdiğiniz ruh hâlini özellikle **anksiyete, korku ve zihinsel dağınıklığı (disorganization)** 
azaltacak şekilde, daha **coherent / bütün**, **daha da tatlı ve pürüzsüz** bir lo-fi pop parçasına dönüştürür.

Zaman yapısı:
- **İlk %20:** Mevcut dağınıklık ve huzursuzluğun daha belirgin olduğu bölüm
- **Orta %20:** Geçiş fazı (dağınıklık, vibrato ve uyumsuz notalar kademeli azalır)
- **Son %60:** Normalleşme fazı – çok yavaş, çok yumuşak, davulları hafif, lead'i sakin,
  pad ve basın sarıcı olduğu, özellikle rahatlatıcı bölüm

Genel ses yapısı:
- Tempo yavaş
- Daha sine ağırlıklı, üst frekansları törpülenmiş, daha kremsi tonlar
- Hafif vinil dokusu, güçlü low-pass, yumuşak reverb
"""
)

col1, col2 = st.columns(2)
with col1:
    anxious = st.slider("Kaygılı", 0.0, 10.0, 5.0, 0.5)
    depressed = st.slider("Çökkün", 0.0, 10.0, 5.0, 0.5)
    frightened = st.slider("Korkulu", 0.0, 10.0, 5.0, 0.5)
with col2:
    disorganized = st.slider("Zihinsel dağınık", 0.0, 10.0, 7.0, 0.5)
    hopeful = st.slider("Umutlu", 0.0, 10.0, 3.0, 0.5)

duration = st.slider(
    "Müziğin süresi (saniye)",
    min_value=10,
    max_value=120,
    value=45,
    step=5,
    help="10 saniyeden 2 dakikaya kadar seçebilirsiniz.",
)

raw_mood = {
    "anxious": anxious,
    "depressed": depressed,
    "frightened": frightened,
    "disorganized": disorganized,
    "hopeful": hopeful,
}

if st.button("🎧 Ekstra Tatlı & Pürüzsüz Lo-fi Müzik Üret", type="primary"):
    with st.spinner("Müzik üretiliyor (yavaş, ekstra tatlı & pürüzsüz lo-fi)..."):
        wav_bytes, info = generate_normalizing_music(raw_mood, duration_s=duration, sr=44100)

    st.markdown("### ▶️ Normalleştirici Ekstra Tatlı Lo-fi Müzik")
    st.audio(wav_bytes, format="audio/wav")

    st.write(f"🎨 Bu seferki stil: **{info['style_desc']}**")

    st.download_button(
        label="💾 Müziği indir (WAV)",
        data=wav_bytes,
        file_name="lofi_coherence_extra_sweet_slow_music.wav",
        mime="audio/wav",
    )

    st.success(
        "Başta daha dağınık, orta kısımda geçiş, son %60'ında ise özellikle "
        "yatıştırıcı, daha da tatlı ve pürüzsüz "
        "bir lo-fi parça üretildi."
    )

st.caption("Bu araç deneysel ve destekleyicidir.")