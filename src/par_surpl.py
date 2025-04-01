import multiprocessing
from http.cookiejar import debug

import math
import random
import time
from dataclasses import dataclass, field
from fractions import Fraction
from statistics import mean, stdev

import numpy as np
from alive_progress import alive_it
from math import floor, log2
from matplotlib import font_manager
from numpy.f2py.crackfortran import verbose
from rich import print


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

    waits: int = field(default=0)
    log_waits: list[int] = field(default_factory=list)
    lengths: int = field(default=0)
    log_lengths: list[int] = field(default_factory=list)

    def generate(self, i: int, rs):
        while True:
            wait = 0
            wait += random.randrange(self.wFixMin, self.wFixMax)
            # print(self.wK > 0, rs[i] != 0, self.lengths / rs[i], self.load)
            if self.wK > 0 and rs[i] != 0:
                # print(wait, end=' ')
                wait += int(
                    random.randint(1, self.wK) / (self.wK + 1) * 2
                    * (self.lengths / self.load - rs[i])
                )
                # print(wait)

            wait = int(max(0, wait))
            self.waits += wait
            self.log_waits.append(wait)

            length = random.randrange(self.lMin, self.lMax)
            self.lengths += length
            self.log_lengths.append(length)

            full, remainder = divmod(length, self.limit)
            if full > 0:
                yield Transaction(wait, self.limit)
                yield from (Transaction(0, self.limit) for _ in range(full - 1))
                if remainder > 0:
                    yield Transaction(0, remainder)
            else:
                yield Transaction(wait, length)

    def bandwidth(self, delays: int = 0):
        return self.lengths / (self.lengths + self.waits + delays)


@dataclass()
class Log:
    min: int
    max: int
    mean: float
    std: float
    sum: int

    @staticmethod
    def from_values(values):
        return Log(
            min(values),
            max(values),
            mean(values),
            stdev(values),
            sum(values),
        )


@dataclass()
class Results:
    counter: int
    last_response: int
    delays: Log
    waits: Log
    lengths: Log
    budget: int
    period: int

    @property
    def up_bandwidth(self) -> float:
        return self.lengths.sum / (self.lengths.sum + self.waits.sum)

    @property
    def down_bandwidth(self) -> float:
        return self.lengths.sum / (self.lengths.sum + self.waits.sum + self.delays.sum)

    # @property
    # def target_bandwidth(self) -> float:
    #     return self.budget / self.period

    # @property
    # def error_bandwidth(self) -> float:
    #     return -(self.target_bandwidth - self.down_bandwidth - max(0, self.up_bandwidth - self.target_bandwidth))


def simulate(traffics, budgets, downstream: Fraction = 1,
             /, *,
             title: str = '',
             steps: int = int(1e5),
             debug: bool = False,
             verbose: bool = False):
    debug = print if debug else lambda _: None

    # Input data.
    responses = np.zeros(len(traffics), dtype=int)
    traffics_iter = [traffic.generate(i, responses) for i, traffic in enumerate(traffics)]
    transactions = [next(t) for t in traffics_iter]
    counter = np.zeros_like(transactions)

    current = 0

    # State..
    budgets = np.array(budgets)
    available = np.zeros_like(budgets, dtype=int)

    offset = 0
    should_incr = available < budgets
    period = budgets[should_incr].sum() * downstream.denominator // downstream.numerator
    period = period if period != 0 else 1
    available[should_incr] += budgets[should_incr]
    accounted = np.zeros_like(budgets, dtype=bool)

    # ## Initialize
    for i in range(len(budgets)):
        if transactions[i].wait == 0 and not accounted[i]:
            if available[i] > 0:
                available[i] -= transactions[i].length
                accounted[i] = True

    # Logs
    ## Per transaction
    log_delays = [list((0,)) for _ in range(len(traffics_iter))]
    ## Measure
    last_lengths = np.zeros_like(budgets, dtype=float)
    avg_waits = np.zeros_like(budgets, dtype=float)
    avg_period = 0
    num_periods = 1

    progress = alive_it if verbose else lambda it, *_args, **_kwargs: it
    for _ in progress(range(steps), title=title):
        debug(f'@ ({offset}, {period})')
        debug(f'  W{[t.wait for t in transactions]} L{[t.length for t in transactions]}')
        debug(f'  {available} {accounted}')

        # Compute step time.
        min_wait = min([math.inf, *map(lambda t: t.wait, filter(lambda t: t.wait != 0, transactions))])
        ready = [*filter(lambda d: accounted[d[0]] and d[1].wait == 0, enumerate(transactions))]
        min_ready_length = min([math.inf, *map(lambda d: d[1].length, ready)])
        period_left = period - offset if period != 0 else math.inf
        forward_by = min(period_left, min_wait, min_ready_length)
        debug(f'  Forward for {forward_by} ({period_left}, {min_wait}, {min_ready_length})')

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
                responses[i] = current + forward_by
                transactions[i] = next(traffics_iter[i])
                counter[i] += 1
                accounted[i] = False

                log_delays[i].append(0)

        # Absolute time
        current += forward_by

        # End of period
        offset += forward_by
        if offset == period:
            offset = 0

            # print((budgets - available), avg_remaining, num_periods, period)
            # period = period if period != 0 else 1
            # unused = np.zeros_like(available, dtype=int)
            # unused[available == budgets] = period
            # avg_lengths += ((period - available) - avg_lengths) / num_periods
            # avg_waits += (period * (available > 0) - avg_waits) / num_periods
            # avg_waits += ((last_lengths - available) - avg_waits) / num_periods
            # last_lengths = available
            # avg_period += (period - avg_period) / num_periods
            # num_periods += 1

            should_incr = available < budgets
            period = budgets[should_incr].sum() * downstream.denominator // downstream.numerator
            period = period if period != 0 else 1
            available[should_incr] += budgets[should_incr]
            available = np.min(np.stack((available, budgets)), axis=0)
            debug(f'  New period: {period}')

        for i in range(len(budgets)):
            if transactions[i].wait == 0 and not accounted[i]:
                if available[i] > 0:
                    available[i] -= transactions[i].length
                    accounted[i] = True

    if verbose:
        # print(
        #     f'Measure: {avg_period}\n'
        #     f'         {avg_period / sum(budgets) * 100:.5f}%\n'
        #     f'         {budgets - avg_waits}\n'
        #     f'         {avg_waits - budgets / avg_period}\n'
        # )

        for i in range(len(budgets)):
            print(
                f'  Traffic #{i}: D{mean(log_delays[i]):.5f} {responses[i]} {counter[i]}\n'
                f'              {traffics[i].bandwidth() * 100:.5f}% {traffics[i].bandwidth(sum(log_delays[i])) * 100:.5f}%\n'
                f'              {mean(traffics[i].log_waits):.5f} {stdev(traffics[i].log_waits):.5f} [{min(traffics[i].log_waits)}, {max(traffics[i].log_waits)}]'
            )

    return [
        Results(
            int(counter[i]), int(responses[i]),
            Log.from_values(log_delays[i]),
            Log.from_values(traffics[i].log_waits),
            Log.from_values(traffics[i].log_lengths),
            int(budgets[i]),
            int(sum(budgets) // downstream),
        )
        for i in range(len(budgets))
    ]


def main():
    # three_way(1)
    # one_two(200, debug=True, steps=250)
    # one_two_uneven(200, debug=True, steps=250)
    # three_halves_on_half(1)
    # three_halves_on_two(2)
    # fibo_ratio(1)
    # bursty_unlimited(1)
    # no_feedback(1)
    # no_feedback_critical(1)
    # test(1, verbose=True)
    # test(8)
    # test(16)
    # test(31)
    # test(32)
    # test(33)
    # test(34)
    # test(64)
    # test(128)
    # test(1024)

    # return test(32, verbose=True)

    testcase = three_halves_on_weirddown
    xvalues = [*range(1, 64 + 1, 1)]
    values = {k: v for k, v in alive_it(
        multiprocessing.Pool(8).imap(testcase, xvalues),
        total=len(xvalues)
    )}
    print(values[next(iter(values.keys()))])

    from matplotlib import pyplot as plt
    serif_font = font_manager.FontProperties(family='CMU Serif Extra', style='normal',
                                             size=12, weight='normal', stretch='normal')
    sans_serif_font = font_manager.FontProperties(family='CMU Sans Serif', style='normal',
                                                  size=12, weight='normal', stretch='normal')
    fig, (ax) = plt.subplots(1, figsize=[8, 8], dpi=500)
    for i in range(len(next(iter(values.values())))):
        ax.plot(values.keys(), [v[i].down_bandwidth for v in values.values()], label=f"#{i + 1}")
    total_bandwidth = [sum(v[i].down_bandwidth for i in range(len(v))) for v in values.values()]
    ax.plot(values.keys(), total_bandwidth, label=f"Total")

    ax.set_xlabel('$K$', font=serif_font)
    ax.set_xticks([2 ** i for i in range(floor(log2(min(xvalues))), 1 + floor(log2(max(xvalues))))])
    ax.set_xticklabels(ax.get_xticks(), font=sans_serif_font)

    ax.set_ylabel('$B$', font=serif_font)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels((f'{t * 100:.0f}%' for t in ax.get_yticks()), font=sans_serif_font)

    ax.legend(prop=serif_font)
    fig.savefig(f"output/limit-error.pdf")
    plt.show()


def test(K: int, **kwargs):
    traffics = [
        Traffic(4 / 10, 8, 8 + 1, 1, 0, 0 + 1, 4),
        Traffic(4 / 10, 64, 64 + 1, 1, 0, 0 + 1, 4),
    ]
    budgets = [1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(1), title=f'test({K})',
        verbose=False if 'verbose' not in kwargs else kwargs.pop('verbose'), **kwargs
    )


def test_mean_limited(K: int, **kwargs):
    traffics = [
        Traffic(1 / 2, 1, 17, 1, 1, 17, 1),
        Traffic(1 / 2, 1, 33, 1, 1, 33, 1),
    ]
    budgets = [1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(1), title=f'test({K})', **kwargs
    )


def simple_three_way(K: int, **kwargs):
    traffics = [
        Traffic(1 / 5, 4, 16 + 1, 1, 0, 0 + 1, 4),
        # Traffic(1 / 4, 4, 64 + 1, 1, 0, 0 + 1, 4),
        Traffic(1 / 1, 4, 64 + 1, 1, 0, 0 + 1, 4),
    ]
    budgets = [1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(1), title=f'test({K})', **kwargs
    )


def three_way(K: int, **kwargs):
    traffics = [
        Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
    ]
    budgets = [1 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, title=f'three_way({K})', **kwargs
    )


def one_two(K: int, **kwargs):
    traffics = [
        Traffic(25 / 100, 1, 32, 16, 0, 1, 8),
        Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
    ]
    budgets = [1 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, title=f'one_two({K})', **kwargs
    )


def one_two_uneven(K: int, **kwargs):
    traffics = [
        Traffic(25 / 100, 1, 32, 16, 0, 1, 8),
        Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(95 / 100, 1, 32, 4, 0, 1, 8),
    ]
    budgets = [10 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, title=f'one_two_uneven({K})', **kwargs
    )


def three_halves_on_half(K: int, **kwargs):
    traffics = [
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
    ]
    budgets = [1 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(1, 2), title=f'three_halves_on_half({K})', **kwargs
    )


def three_halves_on_weirddown(K: int, **kwargs):
    traffics = [
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
    ]
    budgets = [1 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(7, 8), title=f'three_halves_on_weirddown({K})', **kwargs
    )


def three_halves_on_two(K: int, **kwargs):
    traffics = [
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(50 / 100, 1, 32, 4, 0, 1, 8),
    ]
    budgets = [1 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(2), title=f'three_halves_on_two({K})', **kwargs
    )


def fibo_ratio(K: int, **kwargs):
    traffics = [
        Traffic(10 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(10 / 100, 1, 32, 4, 0, 1, 8),
        Traffic(10 / 100, 1, 32, 4, 0, 1, 8),
    ]
    budgets = [2 * K, 3 * K, 5 * K]

    return K, simulate(
        traffics, budgets, Fraction(1), title=f'fibo_ratio({K})', **kwargs
    )


def bursty_unlimited(K: int, **kwargs):
    traffics = [
        Traffic(10 / 100, 1, 32, 128, 0, 1, 8),
        Traffic(10 / 100, 1, 32, 64, 0, 1, 8),
        Traffic(10 / 100, 1, 32, 32, 0, 1, 8),
        Traffic(10 / 100, 1, 32, 16, 0, 1, 8),
        Traffic(10 / 100, 1, 32, 1, 0, 1, 8),
    ]
    budgets = [1 * K, 1 * K, 1 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(1), title=f'bursty_unlimited({K})', **kwargs
    )


def no_feedback(K: int, **kwargs):
    traffics = [
        Traffic(1, 1, 32, 1, 64, 128, 8),
        Traffic(1, 1, 32, 1, 32, 64, 8),
        Traffic(1, 1, 32, 1, 16, 32, 8),
        Traffic(1, 1, 32, 1, 1, 16, 8),
        Traffic(1, 1, 32, 1, 0, 1, 8),
    ]
    budgets = [1 * K, 1 * K, 1 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(1), title=f'no_feedback({K})', **kwargs
    )


def no_feedback_critical(K: int, **kwargs):
    traffics = [
        Traffic(1, 1, 32, 1, 64, 128, 8),
        Traffic(1, 1, 32, 1, 32, 64, 8),
        Traffic(1, 1, 32, 1, 16, 32, 8),
        Traffic(1, 1, 32, 1, 1, 16, 8),
        Traffic(1, 1, 32, 1, 0, 1, 8),
    ]
    budgets = [4 * K, 1 * K, 1 * K, 1 * K, 1 * K]

    return K, simulate(
        traffics, budgets, Fraction(1), title=f'no_feedback_critical({K})', **kwargs
    )


if __name__ == '__main__':
    main()
