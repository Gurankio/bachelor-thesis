import math
import random
import time
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

import numpy as np
from alive_progress import alive_it
from matplotlib import font_manager


@dataclass()
class Transaction:
    wait: int
    length: int

    def wait_for(self, t: int) -> 'Transaction':
        assert self.wait >= t
        return Transaction(wait=self.wait - t, length=self.length)

    def sent(self, l: int) -> 'Transaction':
        assert self.wait == 0
        assert self.length >= l
        return Transaction(wait=self.wait, length=self.length - l)

    def done(self):
        return self.wait == 0 and self.length == 0


@dataclass()
class Traffic:
    load: float
    lMin: int
    lMax: int
    wK: int
    wFixMin: int
    wFixMax: int
    limit: int

    def generate(self, i: int, rs):
        waits = 0
        lengths = 0
        while True:
            wait = 0
            if lengths / rs[i] > self.load:
                wait = lengths / self.load - rs[i]
                wait = int(max(0, wait))
                wait = random.randint(1, self.wK) / (self.wK + 1) * 2 * wait
            wait += random.randrange(self.wFixMin, self.wFixMax)
            waits += wait

            length = random.randint(self.lMin, self.lMax)
            lengths += length

            full, remainder = divmod(length, self.limit)
            if full > 0:
                yield Transaction(wait, self.limit)
                yield from (Transaction(0, self.limit) for _ in range(full - 1))
                if remainder > 0:
                    yield Transaction(0, remainder)
            else:
                yield Transaction(wait, self.limit)


def simulate(strat, traffics, budgets, period, /, *, title: str = '', steps: int = int(1e5), debug: bool = False):
    debug = print if debug else lambda _: None

    # Input data.
    budgets = np.array(budgets)
    rs = np.zeros(len(traffics))
    limits = np.array([t.limit for t in traffics])
    traffics = [traffic.generate(i, rs) for i, traffic in enumerate(traffics)]
    transactions = [next(t) for t in traffics]

    current = 0
    # Period status.
    offset = 0
    yielded = 0
    # Per transaction status.
    accounted = np.zeros_like(budgets, dtype=bool)
    # Per device status.
    did_yield = np.zeros_like(budgets, dtype=bool)
    remaining = budgets.copy()

    # Logs
    ## Per transaction
    log_delays = [list((0,)) for _ in range(len(traffics))]
    ## Per period
    log_remaining = [list((0,)) for _ in range(len(traffics))]
    total_yields = [list((0,)) for _ in range(len(traffics))]
    total_claims = [list((0,)) for _ in range(len(traffics))]

    for _ in alive_it(range(steps), title=title):
        debug(f'@ ({current}, {offset})')
        debug(f'  Available: {yielded} + {remaining}')

        # Update budgets.
        claims = strat(
            yielded=yielded,
            budgets=budgets,
            remaining=remaining,
            did_yield=did_yield,
            period=period,
            offset=offset,
            limits=limits
        )
        for i in range(len(budgets)):
            if transactions[i].wait == 0 and not accounted[i]:
                if remaining[i] <= 0 and not did_yield[i]:
                    if (claim := claims[i]) > 0:
                        debug(f'    Traffic #{i}: RECLAIM {claim}')
                        yielded -= claim
                        remaining[i] += claim
                        total_claims[i][-1] += int(claim)

                if remaining[i] > 0:
                    remaining[i] -= transactions[i].length
                    accounted[i] = True

        debug(f'  Accounted: {accounted}')
        debug(f'  Remaining: {yielded} + {remaining}')

        # Compute least.
        min_wait = min([math.inf, *map(lambda t: t.wait, filter(lambda t: t.wait != 0, transactions))])
        ready = filter(lambda d: accounted[d[0]] and d[1].wait == 0, enumerate(transactions))
        min_ready_length = min([math.inf, *map(lambda d: d[1].length, ready)])
        forward_by = min(period - offset, min_wait, min_ready_length)
        debug(f'  Forward for {forward_by} ({period - offset}, {min_wait}, {min_ready_length})')

        for i in range(len(budgets)):
            if transactions[i].wait != 0:
                transactions[i] = transactions[i].wait_for(forward_by)
            else:
                if accounted[i]:
                    transactions[i] = transactions[i].sent(forward_by)
                else:
                    debug(f'    Traffic #{i}: DELAY {forward_by}')
                    log_delays[i][-1] += int(forward_by)

        for i in range(len(budgets)):
            if transactions[i].done():
                debug(f'    Traffic #{i} DONE')
                rs[i] = current + forward_by
                transactions[i] = next(traffics[i])
                accounted[i] = False

                log_delays[i].append(0)

        ## Restore budgets if needed
        current += forward_by
        offset += forward_by
        if offset == period:
            debug(f'  Period over.')

            for i in range(len(budgets)):
                log_remaining[i].append(int(remaining[i]))

            offset = 0
            yielded = 0
            remaining = budgets.copy()

            for i in range(len(budgets)):
                # Reset per-device status.
                did_yield[i] = False
                total_yields[i].append(0)
                total_claims[i].append(0)

                if transactions[i].wait > 0 and strat != strat_disabled:
                    debug(f'    Traffic #{i} YIELD')
                    did_yield[i] = True
                    yielded += remaining[i]
                    total_yields[i][-1] += int(remaining[i])
                    remaining[i] = 0

    # Make sure there is no budget being created.
    tys = sum(sum(total_yields[i]) for i in range(len(budgets)))
    tcs = sum(sum(total_claims[i]) for i in range(len(budgets)))
    assert tys - tcs >= 0

    waste = (tys - tcs) / (current // period)
    print(f'"Wasted": {tys - tcs}, {waste:.5f} per period')

    for i in range(len(budgets)):
        print(
            f'  Traffic #{i}: D{mean(log_delays[i]):.5f}'
            f' -{mean(total_yields[i]):.5f} +{mean(total_claims[i]):.5f}'
            f' E{mean(log_remaining[i]):.5f} E%{(budgets[i] - mean(log_remaining[i])) / budgets[i]:.5f}'
        )

    return title, current // period, waste, [
        (i, mean(log_delays[i]), mean(total_yields[i]), mean(total_claims[i]), mean(log_remaining[i]))
        for i in range(len(budgets))
    ]


def save_results(testcase, results):
    from pathlib import Path
    file = (Path(__file__).resolve().parent / 'results.pyl')
    file.touch(exist_ok=True)
    file.write_text(
        file.read_text()
        + repr((time.time_ns(), testcase, *results)) + '\n'
    )

    return results


def strat_disabled(remaining, **_):
    return np.zeros_like(remaining)


def strat_max(period: int, offset: int, remaining, did_yield, yielded: int, **_):
    requests = (remaining <= 0).sum() - did_yield.sum()
    if requests > 0:
        available = min(yielded, (period - offset) * requests) // requests
    else:
        available = 0
    return [available if not dy and r <= 0 else 0 for r, dy in zip(remaining, did_yield)]


def strat_fixed(K: int):
    def inner(period: int, offset: int, remaining, did_yield, yielded: int, **_):
        requests = (remaining <= 0).sum() - did_yield.sum()
        if requests > 0:
            available = min(yielded, min(period - offset, K) * requests) // requests
        else:
            available = 0
        return [available if not dy and r <= 0 else 0 for r, dy in zip(remaining, did_yield)]

    inner.__name__ = f'strat_fixed({K})'
    return inner


def strat_prop(perc: Fraction):
    def inner(period: int, offset: int, remaining, did_yield, yielded: int, **_):
        requests = (remaining <= 0).sum() - did_yield.sum()
        if requests > 0:
            available = min(yielded, (period - offset) * perc * requests) // requests
        else:
            available = 0
        return [available if not dy and r <= 0 else 0 for r, dy in zip(remaining, did_yield)]

    inner.__name__ = f'strat_prop({perc!s})'
    return inner


def strat_max_static(period: int, offset: int, remaining, did_yield, yielded: int, **_):
    requests = np.logical_and(remaining <= 0, ~did_yield)
    claims = np.zeros_like(remaining)
    for i, r in enumerate(requests):
        if r:
            claims[i] = min(yielded, (period - offset))
            yielded -= claims[i]

    return claims


def strat_fix_error_static(remaining, did_yield, yielded: int, **_):
    requests = np.logical_and(remaining <= 0, ~did_yield)
    claims = np.zeros_like(remaining)
    for i, r in enumerate(requests):
        if r:
            claims[i] = min(yielded, -remaining[i])
            yielded -= claims[i]
    return claims


def strat_fix_error_limit_static(remaining, did_yield, limits, yielded: int, **_):
    requests = np.logical_and(remaining <= 0, ~did_yield)
    claims = np.zeros_like(remaining)
    for i, r in enumerate(requests):
        if r:
            claims[i] = min(yielded, -remaining[i])
            yielded -= claims[i]
    for i, (r, limit) in enumerate(zip(requests, limits)):
        if r:
            available = limit if yielded >= limit else 0
            claims[i] += available
            yielded -= available
    return claims


def strat_fix_error_max_static(period, offset, remaining, did_yield, yielded: int, **_):
    requests = np.logical_and(remaining <= 0, ~did_yield)
    claims = np.zeros_like(remaining)
    for i, r in enumerate(requests):
        if r:
            claims[i] = min(yielded, -remaining[i])
            yielded -= claims[i]
    for i, r in enumerate(requests):
        if r:
            available = min(yielded, (period - offset))
            claims[i] += available
            yielded -= available
    return claims


def strat_fix_error_1_whlgt4lim_static(period, offset, remaining, did_yield, limits, yielded: int, **_):
    requests = np.logical_and(remaining <= 0, ~did_yield)
    claims = np.zeros_like(remaining)
    for i, r in enumerate(requests):
        if r:
            claims[i] = min(yielded, -remaining[i])
            yielded -= claims[i]
    for i, (r, limit) in enumerate(zip(requests, limits)):
        if r and period - offset >= limit * 4:
            available = min(yielded, 1)
            claims[i] += available
            yielded -= available
    return claims


def plot(testcase, period, budgets, results):
    from matplotlib import pyplot as plt
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(8, 12))

    num_strats = len(results)
    num_traffic = len(results[0][3])

    serif_font = font_manager.FontProperties(family='CMU Serif', style='normal',
                                             size=12, weight='light', stretch='normal')
    sans_serif_font = font_manager.FontProperties(family='CMU Sans Serif', style='normal',
                                                  size=12, weight='normal', stretch='normal')

    ind = np.arange(0, num_strats)
    print(ind)
    colors = ['#FFC07A', '#77DDDD', '#BAEFC9']
    width = 1 / (num_traffic + 1)

    # title, current // period, waste, [
    #     (i, mean(log_delays[i]), mean(total_yields[i]), mean(total_claims[i]), mean(log_remaining[i]))
    #     for i in range(len(budgets))
    # ]

    for i, color in zip(ind, colors):
        delays = [
            delay
            for strat, periods, waste, traffics in results
            for index, delay, yields, claims, rems in traffics
            if index == i
        ]
        a.barh((num_strats - 1) - ind + width * (num_traffic - i), delays, width, color=color, label=f'#{i}')

        rems = [
            max(0, (budget - rems) / budget - 1)
            for strat, periods, waste, traffics in results
            for (index, delay, yields, claims, rems), budget in zip(traffics, budgets)
            if index == i
        ]
        b.barh((num_strats - 1) - ind + width * (num_traffic - i), rems, width, color=color, label=f'#{i}')

        yields = [
            -yields / budget
            for strat, periods, waste, traffics in results
            for (index, delay, yields, claims, rems), budget in zip(traffics, budgets)
            if index == i
        ]
        c.barh((num_strats - 1) - ind + width * (num_traffic - i), yields, width, color=color, label=f'#{i}')

        claims = [
            claims / budget
            for strat, periods, waste, traffics in results
            for (index, delay, yields, claims, rems), budget in zip(traffics, budgets)
            if index == i
        ]
        c.barh((num_strats - 1) - ind + width * (num_traffic - i), claims, width, color=color)

    strats = [
        strat
        for strat, periods, waste, traffics in results
    ]
    ticks = num_strats - 1 - ind + width * (num_traffic // 2)
    ylim = [-width, num_strats]
    rot = 0

    a.set_title(f'Ritardo', font=serif_font)
    a.set(yticks=ticks, ylim=ylim)
    a.set_yticklabels(strats, font=serif_font, rotation=rot)
    a.set(xticks=a.get_xticks())
    a.set_xticklabels(a.get_xticks(), font=sans_serif_font)
    a.legend(prop=serif_font)
    delay_mean = [
        mean(delay for index, delay, yields, claims, rems in traffics)
        for strat, periods, waste, traffics in results
    ]
    for i, dm in enumerate(delay_mean):
        # Draw a short horizontal dashed line centered at the average value.
        # Adjust the line length (here 1 unit total) as needed.
        lw = width * num_traffic / 2
        a.vlines(dm, ymin=ticks[i] - lw, ymax=ticks[i] + lw,
                 colors='black', linestyles='--', linewidth=1)
        # Place a text label just to the right of the average marker.
        space = (a.get_xticks()[1] - b.get_xticks()[0]) * 3 / 8
        a.text(dm + space, ticks[i], f'{dm:.2f}',
               va='center', ha='center', color='black', font=sans_serif_font)

    b.set_title(f'Errore%', font=serif_font)
    b.set(yticks=ticks, ylim=ylim)
    b.set_yticklabels(strats, font=serif_font, rotation=rot)
    b.set(xticks=b.get_xticks())
    b.set_xticklabels([f'{v * 100:4.1f}%' for v in b.get_xticks()], font=sans_serif_font)
    b.legend(prop=serif_font)
    rems_avgs = [
        mean(max(0, (budget - rems) / budget - 1)
             for (index, delay, yields, claims, rems), budget in zip(traffics, budgets))
        for strat, periods, waste, traffics in results
    ]
    for i, w in enumerate(rems_avgs):
        # Draw a short horizontal dashed line centered at the average value.
        # Adjust the line length (here 1 unit total) as needed.
        lw = width * num_traffic / 2
        b.vlines(w, ymin=ticks[i] - lw, ymax=ticks[i] + lw,
                 colors='black', linestyles='--', linewidth=1)
        # Place a text label just to the right of the average marker.
        space = (b.get_xticks()[1] - b.get_xticks()[0]) / 2
        b.text(w + space, ticks[i], f'{w * 100:.2f}%',
               va='center', ha='center', color='black', font=sans_serif_font)

    c.set_title(f'Yields / Claims', font=serif_font)
    c.set(yticks=ticks, ylim=ylim)
    c.set_yticklabels(strats, font=serif_font, rotation=rot)
    c.set(xticks=c.get_xticks())
    c.set_xticklabels([f'{v * 100:4.1f}%' for v in c.get_xticks()], font=sans_serif_font)
    c.legend(prop=serif_font)
    wastes = [
        -waste / sum(budgets)
        for strat, periods, waste, traffics in results
    ]
    wastes = [
        waste if waste != 0 else 0
        for waste in wastes
    ]
    for i, w in enumerate(wastes):
        # Draw a short horizontal dashed line centered at the average value.
        # Adjust the line length (here 1 unit total) as needed.
        lw = width * num_traffic / 2
        c.vlines(w, ymin=ticks[i] - lw, ymax=ticks[i] + lw,
                 colors='black', linestyles='--', linewidth=1)
        # Place a text label just to the right of the average marker.
        space = abs(c.get_xticks()[1] - c.get_xticks()[0]) / 2
        c.text(w + space * (1 if w >= 0 else -1), ticks[i], f'{w * 100:.2f}%',
               va='center', ha='center', color='black', font=sans_serif_font)

    plt.tight_layout()
    plt.savefig(f'output/{testcase}.pdf')
    plt.show()


def main():
    strat_disabled.__name__ = 'Disabilitata'
    strat_fix_error_static.__name__ = 'Errore (Statico)'
    strat_max.__name__ = 'Massimo'
    strat_max_static.__name__ = 'Massimo (Statico)'
    strat_fix_error_limit_static.__name__ = 'Errore + Fisso(Limite) (Statico)'
    strat_fix_error_max_static.__name__ = 'Errore + Massimo (Statico)'
    strat_fix_error_1_whlgt4lim_static.__name__ = 'Errore + 1 se L>4 (Statico)'

    strats = [
        strat_disabled,
        strat_max,
        strat_max_static,
        strat_fix_error_static,
        strat_fix_error_limit_static,

        # strat_fixed(1),
        # strat_fixed(4),

        # Good ones:

        strat_fix_error_max_static,
        strat_fix_error_1_whlgt4lim_static,

        # strat_fixed(16),
        # strat_fixed(32),
        # strat_prop(Fraction(1, 8)),
        # strat_prop(Fraction(1, 4)),
        # strat_prop(Fraction(1, 2)),
        # strat_prop(Fraction(2, 3)),
    ]

    def one_two(K: int):
        traffics = [
            Traffic(5 / 100, 1, 32, 4, 0, 1, 8),
            Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
            Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
        ]
        period = 3 * K
        budgets = [1 * K, 1 * K, 1 * K]

        results = []
        for strat in strats:
            print(testcase := f'one_two({K})')
            results.append(save_results(testcase, simulate(
                strat, traffics, budgets, period, title=f'{strat.__name__}'
            )))
        return testcase, period, budgets, results

    def three_halves_on_two(K: int):
        traffics = [
            Traffic(50 / 100, 1, 32, 4, 1, 16, 8),
            Traffic(50 / 100, 1, 32, 4, 1, 16, 8),
            Traffic(50 / 100, 1, 32, 4, 1, 16, 8),
        ]
        period = 3 * K
        budgets = [1 * K, 1 * K, 1 * K]

        results = []
        for strat in strats:
            print(testcase := f'three_halves_on_two({K})')
            results.append(save_results(testcase, simulate(
                strat, traffics, budgets, period, title=f'{strat.__name__}'
            )))
        return testcase, period, budgets, results

    def one_three_on_half(K: int):
        traffics = [
            Traffic(95 / 100, 1, 32, 4, 1, 16, 8),
            Traffic(5 / 100, 1, 32, 4, 1, 16, 8),
            Traffic(5 / 100, 1, 32, 4, 1, 16, 8),
            Traffic(5 / 100, 1, 32, 4, 1, 16, 8),
        ]
        period = 12 * K
        budgets = [3 * K, 1 * K, 1 * K, 1 * K]

        results = []
        for strat in strats:
            print(testcase := f'one_three_on_half({K})')
            results.append(save_results(testcase, simulate(
                strat, traffics, budgets, period, title=f'{strat.__name__}'
            )))
        return testcase, period, budgets, results

    plot(*one_two(4))
    plot(*one_two(16))
    plot(*one_two(64))
    # one_three_on_half(4)
    plot(*one_three_on_half(64))
    # one_three_on_half(64)
    # five_halves_on_two(1)
    # five_halves_on_two(4)
    # five_halves_on_two(16)
    # five_halves_on_two(64)


if __name__ == '__main__':
    main()
