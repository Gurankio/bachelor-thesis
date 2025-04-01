import random
import textwrap
from dataclasses import dataclass
from typing import Generator

import kneed
import numpy as np
import scipy.stats
from alive_progress import alive_it
from matplotlib import pyplot as plt, font_manager
from rich import print
from scipy import cluster


def make_data_and_ranges(lengths, waits):
    data = np.stack((lengths, waits), axis=-1)
    values, counts = np.unique(data, return_counts=True, axis=0)
    perc = counts / counts.sum()

    ml, Ml = lengths.min(), lengths.max()
    mw, Mw = waits.min(), waits.max()

    data = np.zeros((Ml - ml + 1, Mw - mw + 1))
    for i, p in zip(values, perc):
        l, w = i
        data[l - ml, w - mw] = p

    return data, ml, Ml, mw, Mw


def plot_dist_heatmap(title, data, ml, Ml, mw, Mw):
    """
    :param data: (lengths, waits)
    """
    fig, ax = plt.subplots(dpi=500)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#FFFFFF', "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9"]  # Red, yellow, green, blue
    colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9"]  # Red, yellow, green, blue
    cmap_pastel = LinearSegmentedColormap.from_list("PastelRainbow", colors)
    im = ax.imshow(data, cmap=cmap_pastel)

    plt.rcParams.update({
        "text.usetex": True,
        # 'font.family': 'serif',
        # 'font.serif': 'CMU Serif',
        "mathtext.fontset": "cm",
    })
    # plt.rcParams['text.latex.preamble'] = textwrap.dedent(r'''
    # ''')

    serif_font = font_manager.FontProperties(family='CMU Serif Extra', style='normal',
                                             size=14, weight='normal', stretch='normal')
    sans_serif_font = font_manager.FontProperties(family='CMU Sans Serif', style='normal',
                                                  size=12, weight='normal', stretch='normal')
    # Show all ticks and label them with the respective list entries
    ax.set_xlabel('$A_i$', font=serif_font)
    ax.set_xticks(range(Mw - mw + 1), labels=[f'{o + mw}' for o in range(Mw - mw + 1)])

    ax.set_ylabel('$L_i$', font=serif_font)
    ax.set_yticks(range(Ml - ml + 1), labels=[f'{r + ml}' for r in range(Ml - ml + 1)])

    # Loop over data dimensions and create text annotations.
    for i in range(Ml - ml + 1):
        for j in range(Mw - mw + 1):
            # Auto-adjust text color for visibility
            rgb = im.cmap(im.norm(data[i, j]))[:3]
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = 'black' if luminance > 0.5 else 'white'

            ax.text(j, i, f'{data[i, j] * 100:.2f}' if data[i, j] * 100 >= 0.1 else '',
                    ha="center", va="center", color=text_color, family=sans_serif_font.get_family())

    ax.set_title(title, family="CMU Serif")
    fig.savefig(f"output/{title}.pdf")
    plt.show()
    plt.close()


@dataclass()
class Transaction:
    wait: int
    length: int


def main():
    # Traffic configuration
    # Assumes uniform distribution
    traffic_beats = 1, 17
    print(f'lMin={traffic_beats[0]}, lMax={traffic_beats[1]}')

    # traffic_waitK = 0
    # print(f'[yellow]traffic K[/]={traffic_waitK:d}')
    # traffic_load = Fraction(1, 1)
    # if traffic_waitK > 0:
    #     print(f'[yellow]traffic throughput%(ideal)[/]={traffic_load:f}')

    traffic_wFixMin, traffic_wFixMax = 1, 17
    print(f'wFixMin={traffic_wFixMin}, wFixMax={traffic_wFixMax}')

    # Splitter configuration
    limit = 8
    print(f'{limit=:d}')

    # Isolate configuration
    K = limit + traffic_wFixMax - 1
    budget = K * 1
    period = K * 3
    print(f'([yellow]budget[/], [yellow]period[/])={(budget, period)}')

    # Log arrival to memory
    response = []

    # Behavior
    def traffic(transactions) -> Generator[Transaction, None, None]:
        length = random.randrange(*traffic_beats)
        traffic.sum += length
        yield Transaction(0, length)

        for _ in range(1, transactions):
            # current_ratio = Fraction(traffic.sum, response[-1])
            # if current_ratio - traffic_load > 0:
            #     # Did too much, need to wait
            #     wait = int(max(0, int(current_ratio.numerator / traffic_load - current_ratio.denominator)))
            #     wait = (random.randint(1, traffic_waitK) if traffic_waitK != 0 else 0) / (traffic_waitK + 1) * 2 * wait
            wait = random.randrange(traffic_wFixMin, traffic_wFixMax)
            length = random.randrange(*traffic_beats)
            traffic.sum += length
            yield Transaction(wait, length)

    traffic.sum = 0
    traffic.waits = []
    traffic.lengths = []
    traffic.starts = []
    traffic.ends = []

    def splitter(upstream: Transaction) -> Generator[Transaction, None, None]:
        # Does not add any delay at all, given that there are less than K (synth parameter) concurrent transactions.
        full, remainder = divmod(upstream.length, limit)
        if full > 0:
            yield Transaction(upstream.wait, limit)
            yield from (Transaction(0, limit) for _ in range(full - 1))
            if remainder > 0:
                yield Transaction(0, remainder)
        else:
            yield upstream

    splitter.waits = []
    splitter.lengths = []

    def buffer(upstream: Transaction) -> Generator[Transaction, None, None]:
        buffer.time += upstream.wait
        empty = buffer.time + upstream.length > buffer.prev_last

        if empty:
            buffer.delay.append(delay := upstream.length - 1)

            start_time = buffer.time + delay
            yield Transaction(start_time - buffer.prev_last, upstream.length)

            # If empty then it will complete after the delay.
            buffer.prev_last = start_time + upstream.length
        else:
            buffer.delay.append(delay := buffer.delay[-1])
            # We already accounted for the delay the first memory_time, it remains constant!
            buffer.prev_last += upstream.length
            #  Also, no wait as they are outputted one after the other.
            yield Transaction(0, upstream.length)

        # Account for this transaction.
        buffer.time += upstream.length

        # Alt impl:
        # prev_done = buffer.delay[-1] if len(buffer.delay) > 0 else 0
        # current_ready = upstream.wait + upstream.length - 1
        #
        # wait_prev = max(prev_done - current_ready, 0)
        # wait_current = upstream.length - 1
        # buffer.delay.append(delay := wait_prev + wait_current)
        #
        # next_wait = max(current_ready - prev_done, 0)
        #
        # # Debug
        # # print(f'({prev_done}) {upstream.wait}, {upstream.lengths} -> {next_wait}, {upstream.lengths}')
        #
        # yield Transaction(next_wait, upstream.length)

    buffer.waits = []
    buffer.lengths = []
    buffer.delay = []
    buffer.time = 0
    buffer.prev_last = 0

    def isolate(upstream: Transaction) -> Generator[Transaction, None, None]:
        nonlocal budget

        def incr_time(value: int):
            nonlocal budget

            isolate.time = isolate.time + value
            while isolate.time >= period:
                isolate.time -= period
                if isolate.budget < 0:
                    isolate.budget += budget
                else:
                    isolate.budget = budget

        # Wait
        incr_time(upstream.wait)

        # Delay
        delay = 0
        while isolate.budget <= 0:
            period_left = period - isolate.time
            incr_time(period_left)
            delay += period_left
        isolate.delays.append(delay)

        # Length
        isolate.budget -= upstream.length
        incr_time(upstream.length)

        yield Transaction(upstream.wait + delay, upstream.length)

    # Isolate State
    isolate.time = 0  # meaningless for high enough N
    isolate.budget = budget
    isolate.periods = []
    isolate.remaining = [0]
    isolate.offsets = []
    isolate.delays = []
    isolate.lengths = []
    isolate.waits = []
    isolate.availables = []

    splitter.starts = []
    splitter.ends = []
    buffer.starts = []
    buffer.ends = []
    isolate.starts = []
    isolate.ends = []

    N = int(2.5 * 1e5)
    traffic_time = 0
    splitter_time = 0
    buffer_time = 0
    isolate_time = 0
    for t in alive_it(traffic(N), total=N):
        traffic.starts.append(traffic_time)
        traffic.waits.append(0)
        traffic.lengths.append(0)
        splitter.starts.append(splitter_time)
        splitter.waits.append(0)
        splitter.lengths.append(0)
        buffer.starts.append(buffer_time)
        buffer.waits.append(0)
        buffer.lengths.append(0)
        isolate.starts.append(isolate_time)
        isolate.waits.append(0)
        isolate.lengths.append(0)

        traffic.waits[-1] += t.wait
        traffic.lengths[-1] += t.length
        traffic_time += t.wait + t.length

        for s in splitter(t):
            splitter.waits[-1] += s.wait
            splitter.lengths[-1] += s.length
            splitter_time += s.wait + s.length

            for b in buffer(s):
                buffer.waits[-1] += b.wait
                buffer.lengths[-1] += b.length
                buffer_time += b.wait + b.length

                # Stats
                isolate.periods.append(isolate.time // period)
                isolate.offsets.append(isolate.time % period)
                isolate.availables.append(isolate.budget)

                for i in isolate(b):
                    isolate.waits[-1] += i.wait
                    isolate.lengths[-1] += i.length
                    isolate_time += i.wait + i.length

        # Timing
        isolate.ends.append(isolate_time)
        buffer.ends.append(buffer_time)
        splitter.ends.append(splitter_time)
        traffic.ends.append(traffic_time)

    print()

    print('[dim]Converting to numpy...')
    traffic_waits = np.array(traffic.waits)
    traffic_lengths = np.array(traffic.lengths)
    traffic_starts = np.array(traffic.starts)
    traffic_ends = np.array(traffic.ends)

    splitter_waits = np.array(splitter.waits)
    splitter_lengths = np.array(splitter.lengths)
    splitter_starts = np.array(splitter.starts)
    splitter_ends = np.array(splitter.ends)

    buffer_waits = np.array(buffer.waits)
    buffer_lengths = np.array(buffer.lengths)
    buffer_starts = np.array(buffer.starts)
    buffer_ends = np.array(buffer.ends)

    isolate_waits = np.array(isolate.waits)
    isolate_lengths = np.array(isolate.lengths)
    isolate_starts = np.array(isolate.starts)
    isolate_ends = np.array(isolate.ends)

    buffer_delays = np.array(buffer.delay)
    isolate_availables = np.array(isolate.availables)
    isolate_offsets = np.array(isolate.offsets)
    isolate_remaining = np.array(isolate.remaining)
    isolate_delays = np.array(isolate.delays)

    response = np.array(response)

    # print('[bold]isolate[/]')
    # total_transactions = np.unique_counts(isolate_periods).counts.sum()
    # assert total_transactions == len(isolate_remaining)
    # total_periods = isolate_periods.max()
    # trans_in_period = total_transactions / total_periods
    # print(f'{trans_in_period=!s} trans/period')
    # mean_beats = isolate_lengths.mean()
    # print(f'{mean_beats=!s} beats/trans')
    # effective_budget = mean_beats * trans_in_period
    # print(f'{effective_budget=!s} beats/period')
    # overbudget = effective_budget - budget
    # print(f'{overbudget=!s} beats/period')
    # if overbudget < 0:
    #     print('  [red][bold]Underused[/bold] or given more budget than actually available!')
    # else:
    #     print('  [green]Isolated successfully [dim](unless period is too small)[/].')
    # print()

    # plot_dist_heatmap('isolate states', *make_data_and_ranges(isolate_remaining, isolate_offsets))
    # plot_dist_heatmap('buffer states', *make_data_and_ranges(buffer_delays, np.zeros_like(buffer_delays)))

    # print('[bold]dumb delays[/]')
    # print(f'{isolate_delays.mean()=!s}')
    # print(f'[magenta]delay (piecewise)[/]={buffer_delays.mean() + isolate_delays.mean()!s}')
    # print()

    print('[bold]media ritardi per transazione (E[R_i]) [/]')
    ber = (buffer_ends - buffer_starts - (traffic_waits + traffic_lengths))
    print(f'[blue   ]E[R_i] (buffer  )[/]={ber.mean()!s}')
    ier = (isolate_ends - isolate_starts - (buffer_waits + buffer_lengths))
    print(f'[magenta]Max[R_i] (isolate )[/]={ier.max()!s}')
    print(f'[magenta]E[R_i] (isolate )[/]={ier.mean()!s}')

    ###
    # print('[bold]fit lineare[/]')

    ### Buffer
    # ys = (buffer_ends - buffer_starts - (traffic_waits + traffic_lengths)).cumsum()
    # # ys = buffer_ends - splitter_ends # same thing as above
    #
    # states_hits = buffer_delays
    # states = list(sorted(set([state for state in states_hits])))
    # states_freq = [np.zeros(len(states))]
    # for state in alive_it(states_hits, title='Recreating states... (buffer)'):
    #     prev = states_freq[-1]
    #     prev = prev.copy()
    #     j = states.index(state)
    #     prev[j] += 1
    #     states_freq.append(prev)
    # states_freq = np.array(states_freq[1:])
    # states_freq /= np.arange(1, N + 1)[:, None]
    # states_freq = states_freq - states_freq[-1]
    # states_freq = np.linalg.norm(states_freq, axis=1)
    # kneedle = kneed.KneeLocator(
    #     x=np.arange(0, N),
    #     y=states_freq,
    #     curve="convex",
    #     direction="decreasing",
    # )
    # print(states_freq)
    # K = next(i for i in range(len(states_freq)) if states_freq[i] < 1e-6)
    # K = kneedle.elbow
    # print(f'Stationary after: {K}')
    # m, c = np.diff(ys[K:], 1).mean(), ys[K]
    # buffer_R_T_out = c
    # print(f'[green]E[R_i], E[R_T] (buffer  )[/]=[bold]{m:10.6f}[/], [dim]{c:12.6f}[/]')
    # xs = np.arange(1, len(ys) + 1)
    #
    # A = np.vstack([xs, np.ones(len(xs))]).T
    #
    # def f(x):
    #     tmp = A @ x - ys
    #     return tmp.dot(tmp)
    #
    # res = scipy.optimize.minimize(f, np.ones(2), bounds=[(0, None), (0, None)])
    # m, c = res.x
    # print(f'[blue]E[R_i], E[R_T] (buffer  )[/]=[dim]{m:10.6f}[/], [bold]{c:12.6f}[/]')
    #
    # m, c = np.polyfit(np.arange(1, len(ys) + 1), ys, 1)
    # print(f'[dim][blue   ]E[R_i], E[R_T] (buffer  )[/]=[dim]{m:10.6f}[/], [bold]{c:12.6f}[/]')

    ### Isolate
    # ys = (isolate_ends - isolate_starts - (traffic_waits + traffic_lengths)).cumsum()
    # # ys = isolate_ends - buffer_ends # same thing as above
    #
    # states_hits = np.stack((isolate_availables, isolate_offsets), axis=-1)
    # states = list(sorted(set([tuple(state) for state in states_hits])))
    # states_freq = [np.zeros(len(states))]
    # for state in alive_it(states_hits, title='Recreating states... (isolate)'):
    #     state = tuple(state)
    #     prev = states_freq[-1]
    #     prev = prev.copy()
    #     j = states.index(state)
    #     prev[j] += 1
    #     states_freq.append(prev)
    # states_freq = np.array(states_freq[1:])
    # states_freq /= np.arange(1, N + 1)[:, None]
    # states_freq = states_freq - states_freq[-1]
    # states_freq = np.linalg.norm(states_freq, axis=1)
    # kneedle = kneed.KneeLocator(
    #     x=np.arange(0, N),
    #     y=states_freq,
    #     curve="convex",
    #     direction="decreasing",
    # )
    # print(states_freq)
    # K = next(i for i in range(len(states_freq)) if states_freq[i] < 1e-1)
    # K = kneedle.elbow
    # print(f'Stationary after: {K}')
    # m, c = np.diff(ys[K:], 1).mean(), ys[K]
    # isolate_R_T_out = c
    # print(f'[green]E[R_i], E[R_T] (isolate )[/]=[bold]{m:10.6f}[/], [dim]{c:12.6f}[/]')
    #
    # xs = np.arange(1, len(ys) + 1)
    #
    # A = np.vstack([xs, np.ones(len(xs))]).T
    #
    # def f(x):
    #     tmp = A @ x - ys
    #     return tmp.dot(tmp)
    #
    # res = scipy.optimize.minimize(f, np.ones(2), bounds=[(0, None), (0, None)])
    # m, c = res.x
    # print(f'[magenta]E[R_i], E[R_T] (isolate )[/]=[bold]{m:10.6f}[/], [dim]{c:12.6f}[/]')
    # m, c = np.polyfit(np.arange(1, len(ys) + 1), ys, 1)
    # print(f'[dim][magenta]E[R_i], E[R_T] (isolate )[/]=[bold]{m:10.6f}[/], [dim]{c:12.6f}[/]')
    # print()

    ###
    print('[bold]precisione limite[/]')
    rem = isolate_remaining.mean()
    # print(f'[yellow]E[Remaining][/]={rem:.6f}')
    # print(f'[yellow]effective_budget[/]={budget - rem:.6f}')
    # print(f'[yellow]E[Remaining]%[/]={(budget - rem) / budget:.6f}')
    # print(f'ratio={(budget - rem) / period:.6f}')
    # if rem < 0:
    #     if budget - rem < period:
    #         print('[green]  Isolated![/]')
    #     else:
    #         print('[red]  Did not isolate [dim](effective budget > period)[/dim][/]')
    # else:
    #     print('[red]  Did not isolate [dim](load too little)[/dim][/]')
    print(f'ratio_check={(isolate_lengths.mean() / (isolate_lengths.mean() + isolate_waits.mean())):.6f}')

    def plot_hist():
        fig, (a, b) = plt.subplots(2)
        fig.set_size_inches(6, 8)
        fig.set_dpi(240)
        # fig.tight_layout()

        a.set_title('buffer delays')
        bins = np.arange(0, buffer_delays.max() + 1.5) - 0.5
        a.hist(buffer_delays, bins=bins, rwidth=0.8)

        b.set_title('isolate delays')
        bins = np.arange(0, isolate_delays.max() + 1.5) - 0.5
        b.hist(isolate_delays, bins=bins, rwidth=0.8)

        plt.show()

    def plot_heatmaps():
        # plot_dist_heatmap('traffic', *make_data_and_ranges(traffic_lengths, traffic_waits))
        # plot_dist_heatmap('splitter', *make_data_and_ranges(splitter_lengths, splitter_waits))
        # plot_dist_heatmap('buffer', *make_data_and_ranges(buffer_lengths, buffer_waits))
        # plot_dist_heatmap('isolate', *make_data_and_ranges(isolate_lengths, isolate_waits))
        ...

    # plot_hist()
    # plot_heatmaps()

    # return buffer_R_T_out, isolate_R_T_out


if __name__ == '__main__':
    main()
    # orig_print = print
    # print = lambda *args: None
    # orig_alive_it = alive_it
    # alive_it = lambda it, *args, **kwargs: it
    # samples = np.array([main() for _ in orig_alive_it(range(int(1e3)))], dtype=float)
    # print = orig_print
    # print(samples)
    # print(samples[:, 0].mean(), samples[:, 1].mean())
