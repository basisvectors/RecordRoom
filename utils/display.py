import matplotlib.pyplot as plt
import numpy as np
import time

SAMPLE_RATE = 44100

# TIME UTILS
def convert_seconds_to_mmss(second_ticks: np.ndarray):
    return np.array([time.strftime("%M:%S", time.gmtime(t)) for t in second_ticks])
def samples_to_bars(samples: int, tempo: int, sample_rate: int = 44100):
    return (samples * tempo) / (60 * 4 * sample_rate)
# repeated this way for better clarity in the code
def ms_to_bars(ms: int, tempo: int):
    return ((ms / 1000) * tempo) / (60 * 4)
def bars_to_samples(bars: int, tempo: int, sample_rate: int = 44100):
    return round((bars * 4 * 60 * sample_rate) / (tempo))
# repeated this way for better clarity in the code
def bars_to_ms(bars: int, tempo: int):
    return round((bars * 4 * 60 * 1000) / (tempo))
def boundaries_to_cursors(boundaries, tempo):
    return [
        *[bars_to_samples(b[0], tempo) for b in boundaries],
        bars_to_samples(boundaries[-1][1], tempo),
    ]

#DISPLAY UTILS
def show_audio(
    audio: np.ndarray, tempo, cursors: list = None, axvls: list = None, figsize=(12, 4)
):
    duration = len(audio) / SAMPLE_RATE  # in seconds
    t = np.linspace(0, duration, len(audio))
    bars_per_beat = 4
    beats_per_second = tempo / (60)
    bars_per_second = beats_per_second / bars_per_beat
    # Define tempo and calculate the duration of each bar
    bar_duration = 1 / bars_per_second
    total_bars = duration * bars_per_second

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    # Plot the audio waveform
    ax.plot(t, audio)
    if cursors is not None:
        for index, cursor in enumerate(cursors[:-1]):
            ax.axvspan(
                cursor / SAMPLE_RATE,
                cursors[index + 1] / SAMPLE_RATE,
                color="r" if index % 2 == 0 else "b",
                alpha=0.3,
            )
    if axvls:
        for axvl in axvls:
            ax.axvline(axvl / SAMPLE_RATE, color="g", alpha=0.8)

    # Set the lower x-axis for bars at the given tempo
    bar_ticks = np.arange(0, total_bars + 1)
    bar_positions = bar_ticks * bar_duration
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_ticks.astype(int))
    ax.set_xlabel("Bars at Tempo = {}".format(tempo))
    ax.grid(visible=True)

    # Set the upper x-axis for time in seconds
    seconds_per_bar = 1 / bars_per_second
    seconds_per_tick = 1 / bars_per_second / 4
    total_ticks = total_bars * 4
    time_ticks = np.arange(0, total_ticks + 1)
    time_positions = time_ticks * seconds_per_tick

    ax2 = ax.twiny()
    ax2.set_xticks(np.arange(0, duration + 1))
    ax2.set_xticklabels(
        convert_seconds_to_mmss(np.arange(0, duration + 1)).astype(str), rotation=90
    )
    ax2.set_xlim(ax.get_xlim())

    ax2.set_xlabel("Time (seconds)")
    plt.show()
    return fig, ax