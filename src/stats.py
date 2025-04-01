import enum
from statistics import mean

import math
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from functools import cache
from itertools import chain
from typing import Callable

# import numba
import sympy
import sys
from alive_progress import alive_it
from rich import print
from scipy.optimize import minimize

# Hack for pretty fractions.
Fraction.__repr__ = Fraction.__str__

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from traffic2 import plot_dist_heatmap


def plot_markov_chain(node_color, labels, transition):
    if len(transition) <= 32:
        from matplotlib import pyplot as plt
        import networkx as nx
        plt.figure(figsize=(8, 8))
        graph = nx.from_numpy_array((transition > 0).astype(int), create_using=nx.DiGraph)
        pos = nx.shell_layout(graph)
        # Supports alpha channel.

        nx.draw(
            graph, pos,
            node_color=node_color, node_size=1200,
            edge_color='gray', font_size=12,
        )
        nx.draw_networkx_labels(graph, pos, labels={n: labels[n] for n in graph})
        plt.show()
    else:
        print('Too many nodes, graph skipped.')


def find_irreducible_classes_and_transients(transition):
    # Step 1: Convert the transition matrix into a graph representation
    n = transition.shape[0]
    graph = (transition > 0).astype(int)  # Binary adjacency matrix

    # Step 2: Find strongly connected components (SCCs)
    scc_count, labels = connected_components(csr_matrix(graph), connection='strong')

    # Step 3: Group states into their respective SCCs
    scc_groups = {i: [] for i in range(scc_count)}
    for state, scc_label in enumerate(labels):
        scc_groups[scc_label].append(state)

    # Step 4: Identify closed SCCs and transient states
    closed_sccs = []
    transient_states = set(range(n))

    for scc, states in scc_groups.items():
        # Check if the SCC is closed (no outgoing edges)
        is_closed = True
        for state in states:
            outgoing_states = np.nonzero(graph[state])[0]
            if any(labels[out] != scc for out in outgoing_states):  # If an edge points outside the SCC
                is_closed = False
                break
        if is_closed:
            closed_sccs.append(states)
            transient_states.difference_update(states)

    return closed_sccs, list(transient_states)


@dataclass(frozen=True)
class Transaction:
    wait: int
    lengths: tuple[int, ...]


class Order(Enum):
    AL = enum.auto()
    LA = enum.auto()

    def reverse(self) -> 'Order':
        return {
            Order.AL: self.LA,
            Order.LA: self.AL,
        }[self]


@dataclass()
class Traffic:
    dist: dict[Transaction, Fraction]
    order: Order = field(default=Order.AL)

    def length_range(self) -> tuple[int, int]:
        return (min(map(lambda t: min(t.lengths), self.dist.keys())),
                max(map(lambda t: max(t.lengths), self.dist.keys())))

    def wait_range(self) -> tuple[int, int]:
        return (min(map(lambda t: t.wait, self.dist.keys())),
                max(map(lambda t: t.wait, self.dist.keys())))

    def bandwidth(self, delay: float = 0.0) -> float:
        dist = self.equivalent_singular()
        mean_length = sum(sum(t.lengths) * p for t, p in dist.dist.items())
        mean_wait = sum(t.wait * p for t, p in dist.dist.items())
        return mean_length / (mean_length + mean_wait + delay)

    def equivalent_singular(self) -> 'Traffic':
        dist = {}
        for t, p in self.dist.items():
            nt = Transaction(t.wait, (t.lengths[0],))
            dist.setdefault(nt, 0)
            dist[nt] += p

            for l in t.lengths[1:]:
                nt = Transaction(0, (l,))
                dist.setdefault(nt, 0)
                dist[nt] += p

        total = sum(dist.values())
        return Traffic({t: p / total for t, p in dist.items()}, self.order)

    def plot(self, title: str):
        def make_data_and_ranges():
            ml, Ml = self.length_range()
            mw, Mw = self.wait_range()

            data = np.zeros((Ml - ml + 1, Mw - mw + 1))
            for t, p in self.equivalent_singular().dist.items():
                l, w = t.lengths[0], t.wait
                data[l - ml, w - mw] = p

            return data, ml, Ml, mw, Mw

        plot_dist_heatmap(title, *make_data_and_ranges())

    @staticmethod
    def uniform(wLow: int, wHigh: int, lLow: int, lHigh: int) -> 'Traffic':
        return Traffic({
            Transaction(w, (l,)): Fraction(1, (lHigh - lLow) * (wHigh - wLow))
            for l in range(lLow, lHigh) for w in range(wLow, wHigh)
        })

    def split(self, limit: int) -> 'Traffic':
        assert limit > 0
        assert self.order == Order.AL

        def handle(l: int):
            full, remainder = divmod(l, limit)
            if full > 0:
                yield from (limit for _ in range(full))
            if remainder > 0:
                yield remainder

        dist = {}
        for t, p in self.dist.items():
            nts = Transaction(t.wait, tuple(chain(*(handle(l) for l in t.lengths))))
            dist.setdefault(nts, 0)
            dist[nts] += p

        return Traffic(dist, order=self.order)

    def buffer(self, states, states_prob) -> 'Traffic':
        dist = {}
        for t, p in self.dist.items():
            for start, state_prob in zip(states, states_prob):
                end = max(start - t.wait, max(t.lengths) - 1)
                nt = Transaction(t.wait + (end - start), t.lengths)
                dist.setdefault(nt, 0)
                dist[nt] += p * state_prob

        return Traffic(dist, order=self.order)

    def isolate(self, budget, period, states, states_prob) -> 'Traffic':
        dist = {}
        for t, p in self.dist.items():
            for start, state_prob in zip(states, states_prob):
                available, offset = start
                delay = 0

                # Wait
                offset += t.wait
                if offset >= period:
                    available = budget
                    offset %= period

                # Length
                for l in t.lengths:
                    # Wait till period over.
                    if available <= 0:
                        delay += period - offset
                        available = budget
                        offset = 0

                    # Do transaction.
                    available -= l
                    offset += l
                    if offset >= period:
                        available = budget
                        offset %= period

                nt = Transaction(t.wait + delay, t.lengths)
                dist.setdefault(nt, 0)
                dist[nt] += p * state_prob

        return Traffic(dist, order=self.order)

    def reverse(self) -> 'Traffic':
        raise NotImplementedError()
        # dist = {}
        # for ts0, p0 in self.dist.items():
        #     for ts1, p1 in self.dist.items():
        #         nts = tuple(chain(ts0, ts1))
        #         nts = tuple(Transaction(lengths=ta.lengths, wait=tb.wait) for ta, tb in zip(nts[:-1], nts[1:]))
        #         dist.setdefault(nts, 0)
        #         dist[nts] += p0 * p1
        #
        # return Traffic(dist, self.order.reverse())


class Solver(Enum):
    BRUTE = enum.auto()
    NUMPY = enum.auto()
    SCIPY = enum.auto()
    HOPE = enum.auto()
    SYMPY = enum.auto()

    def solve(self, states: int, transition, /, *, verbose: bool = True):
        match self:
            case Solver.BRUTE:
                power = transition.astype(float)
                for _ in alive_it(range(int(1e5))):
                    power = power @ transition
                print(power)
                return power[0, :]

            case Solver.NUMPY:
                result = None
                mat = transition.astype(float)
                power = 2
                while power < 1e10:
                    tmp = np.linalg.matrix_power(mat, power)[0, :]
                    if np.abs(tmp.sum() - 1) < 1e-6:
                        result = tmp
                        power *= 2
                    else:
                        break

                if verbose:
                    print('Solver stats:')
                    print(math.log2(power), power // 4, result.sum())
                return result

            case Solver.SCIPY:
                transition = csr_matrix(transition.astype(np.float64))

                def f(x):
                    y = x - x @ transition
                    # x - x
                    return y.dot(y)

                if verbose:
                    print('[dim]SciPy launched...')

                constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
                res = minimize(f, np.ones(states),
                               constraints=constraints, bounds=[(0, 1) for _ in range(states)])
                if verbose:
                    print(res)
                    print(f'   bound: {res.x.sum()}')
                return res.x

            case Solver.HOPE:
                import numba
                dtype = np.float64
                transition = np.eye(*transition.shape, dtype=dtype) - transition.astype(dtype)

                @numba.njit(fastmath=True)
                def f(x: np.ndarray):
                    y = x @ transition
                    return y.dot(y)

                if verbose:
                    print('[dim]SciPy launched...')

                constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
                res = minimize(f, np.ones(states),
                               constraints=constraints, bounds=[(0, 1) for _ in range(states)])
                if verbose:
                    print(res)
                    print(f'   bound: {res.x.sum()}')
                return res.x

            case Solver.SYMPY:
                x = sympy.symarray('x', states, positive=True)
                eqs = sympy.simplify(x - x @ transition)
                res = next(iter(sympy.solve([*eqs, sympy.Eq(x.sum(), 1)], x, check=True, dict=True)))
                res = np.array([res[xi] for xi in x])
                return res


def build_reachable_mc[T](starting: T, in_dist: Traffic, state_func: Callable[[T, Transaction], T]):
    queue = [starting]
    states = [starting]
    adjacency = {}
    while len(queue) > 0:
        start = queue.pop()

        for t, p in in_dist.dist.items():
            end = state_func(start, t)

            if end not in states:
                states.append(end)
                queue.insert(0, end)

            adjacency.setdefault((start, end), 0)
            adjacency[start, end] += p

    transition = np.zeros((len(states), len(states)), dtype=Fraction)
    for (start, end), prob in adjacency.items():
        transition[states.index(start), states.index(end)] = prob
    states = np.array(states)

    return states, transition


def buffer_mc(in_dist: Traffic, /, *, solver: Solver = Solver.NUMPY, verbose: bool = True):
    def state_func(start: int, t: Transaction):
        end = max(start - t.wait, t.lengths[0] - 1)
        for l in t.lengths[1:]:
            end = max(end, l - 1)
        return end

    states, transition = build_reachable_mc(0, in_dist, state_func)
    classes, transients = find_irreducible_classes_and_transients(transition)

    if verbose:
        print('Buffer Markov Chain')
        print(f'  Total classes: {len(classes)}')
        print(f'  Total transient states: {len(transients)}')

    # def transient_time():
    #     transients_states = [states[i] for i in transients]
    #     queue = [0]
    #     seen = [0]
    #
    #     adjacency = {}
    #     while len(queue) > 0:
    #         start = queue.pop()
    #
    #         for t, p in in_dist.dist.items():
    #             end = state_func(start, t)
    #             delay = max(start, max(t.lengths) - 1) - start
    #
    #             if end in transients_states:
    #                 if end not in seen:
    #                     seen.append(end)
    #                     queue.insert(0, end)
    #
    #             adjacency.setdefault((start, end), (0, 0))
    #             adjacency[start, end] = (adjacency[start, end][0] + p, adjacency[start, end][1] + delay)
    #
    #     print(adjacency)
    #     return 0
    #
    # print(f'-> Transient delay: {transient_time()}')

    # node_color = [f'#AFAEFF{'44' if i in transients else 'FF'}' for i, _ in enumerate(states)]
    # plot_markov_chain(node_color, [f'{s}' for s in states], transition)

    output_states, output_states_prob, output_delay = None, None, None
    for i, irreducible in enumerate(classes):
        if verbose:
            print(f'[bold][magenta]Class [/magenta]{i + 1}:[/]')

            if len(classes) > 1:
                hs = sympy.symarray('h', len(states))
                eqs = [sympy.Eq(hs[i], 1 if i in irreducible else sum(transition[i, j] * hs[j]
                                                                      for j in range(len(states))))
                       for i in range(len(states))]
                print(f'[dim]Hitting probability: {sympy.solve(eqs)[hs[0]]}')

            if len(transients) > 0:
                ks = sympy.symarray('h', len(states))
                eqs = [sympy.Eq(ks[i], 0 if i in irreducible else 1 + sum(transition[i, j] * ks[j]
                                                                          for j in range(len(states))))
                       for i in range(len(states))]
                print(f'[dim]Mean time to hit: {float(sympy.solve(eqs)[ks[0]]):.3f}')

        mask = np.array([i in irreducible for i in range(len(transition))])
        output_states = class_states = states[mask]
        class_transition = transition[mask][:, mask]

        if verbose:
            print(f'states={class_states!s}')
            print(f'transition=\n{textwrap.indent(str(class_transition), '  ')}')
            np.savetxt(f"output/buffer_{i}.txt", class_transition, fmt="%8s")

        output_states_prob = states_prob = solver.solve(len(class_states), class_transition, verbose=verbose)
        if verbose:
            print(f'{states_prob=!s}')

        # Compute delay
        def delay_map(start: int, t: Transaction):
            return max(start - t.wait, max(t.lengths) - 1)

        output_delay = delay = sum(
            delay_map(state, t) * tp * sp
            for t, tp in in_dist.dist.items()
            for state, sp in zip(class_states, states_prob)
        )
        if verbose:
            print(f'[blue]E[R_T] (buffer  )[/]={delay!s}')
            delay = sum(
                state * sp
                for state, sp in zip(class_states, states_prob)
            )
            print(f'[blue]E[R_T] (buffer  )[/]={delay!s}')

        # Compute time
        def time_map(start: int, t: Transaction):
            state = max(start - t.wait, max(t.lengths) - 1)
            return int(state - start)

        # Se viene negativo è solamente un errore di calcolo, usando sympy infatti funziona e da 0 esatto.
        time = sum(
            time_map(state, t) * tp * sp
            for t, tp in in_dist.dist.items()
            for state, sp in zip(class_states, states_prob)
        )

        ### Till stationary
        transient_delay = 0

        if len(transients) > 0:
            mask = np.array([i in transients for i in range(len(transition))])
            tran_states = states[mask]
            tran_transition = transition[mask][:, mask]
            print(tran_transition)

            tran_hits = np.linalg.inv(np.eye(*tran_transition.shape) - tran_transition.astype(float))
            print(tran_hits[0, :])

            outgoing_delay = [
                sum(delay_map(state, t) * tp for t, tp in in_dist.dist.items())
                for state in tran_states
            ]
            print(outgoing_delay)

            transient_delay += sum(tran_hits[0, i] * outgoing_delay[i] for i in range(len(tran_states)))
            print(f'[dim]  truly transient delay: {transient_delay:.3f}')

        outgoing_delay = [
            sum(delay_map(state, t) * tp for t, tp in in_dist.dist.items())
            for state in class_states
        ]
        print(outgoing_delay)

        Pstep = np.eye(len(class_transition))
        steps = 0
        till_stationary_delay = 0
        while True:
            Pstep = Pstep @ class_transition
            steps += 1

            # Compute total variation distance
            tvd = np.max(0.5 * np.sum(np.abs(Pstep - states_prob), axis=1))
            print(f'[dim]  -> {tvd:.0e}, {Pstep[0, :]}')
            if tvd < 1e-6:
                break

            till_stationary_delay = sum(Pstep[0, i] * outgoing_delay[i] for i in range(len(class_states)))

        print(f'[dim]  mixing time: {steps - 1}')
        print(f'[dim]  till stationary delay: {till_stationary_delay:.6f}')

        transient_delay += till_stationary_delay

        if verbose:
            print(f'[green]E[R_T] (buffer  )[/]={transient_delay:.6f}')

        if verbose:
            print(f'[blue]E[R_i] (buffer  )[/]={time!s}')

    return output_states, output_states_prob, output_delay


def isolate_mc(in_dist: Traffic, budget: int, period: int, /, *, solver: Solver = Solver.NUMPY, verbose: bool = True):
    def state_func(start: tuple[int, int], t: Transaction) -> tuple[int, int]:
        available, offset = start

        # Wait
        offset += t.wait
        if offset >= period:
            available = budget
            offset %= period

        # Length
        for l in t.lengths:
            # Wait till period over.
            if available <= 0:
                available = budget
                offset = 0

            # Do transaction.
            available -= l
            offset += l
            if offset >= period:
                available = budget
                offset %= period

        # Debug
        # print(f'{start} -> ({available}, {offset}), {t}')
        return int(available), int(offset)

    states, transition = build_reachable_mc((budget, 0), in_dist, state_func)
    classes, transients = find_irreducible_classes_and_transients(transition)

    if verbose:
        print(f'Isolate Markov Chain ({budget}/{period})')
        print(f'  Total classes: {len(classes)}')
        print(f'  Total transient states: {len(transients)}')

    # Draw the graph
    node_color = ['#AFAEFF' if (b, o) == (budget, 0) else '#B9F7B6' if b > 0 else '#F1B0AD'
                  for b, o in states]
    node_color = [f'{c}{'44' if i in transients else 'FF'}' for i, c in enumerate(node_color)]
    plot_markov_chain(node_color, [f'{s[0]}, {s[1]}' for s in states], transition)

    output_error, output_delay = None, None
    for i, irreducible in enumerate(classes):
        if verbose:
            print(f'[bold][magenta]Class [/magenta]{i + 1}:[/]')

            if len(classes) > 1:
                hs = sympy.symarray('h', len(states))
                eqs = [sympy.Eq(hs[i],
                                1 if i in irreducible else sum(transition[i, j] * hs[j] for j in range(len(states))))
                       for i in range(len(states))]
                print(f'[dim]Hitting probability: {sympy.solve(eqs)[hs[0]]}')

            if len(transients) > 0:
                ks = sympy.symarray('h', len(states))
                eqs = [sympy.Eq(ks[i], 0 if i in irreducible else 1 + sum(transition[i, j] * ks[j]
                                                                          for j in range(len(states))))
                       for i in range(len(states))]
                mean_time_to_hit = float(sympy.solve(eqs)[ks[0]])
                print(f'[dim]Mean time to hit: {mean_time_to_hit:.3f}')

        mask = np.array([i in irreducible for i in range(len(transition))])
        output_states = class_states = states[mask]
        class_transition = transition[mask][:, mask]

        if verbose:
            print(f'states[{len(class_states)}]=\n{textwrap.indent(str(class_states), '  ')}')
            print(f'[yellow]transition[/]=\n{textwrap.indent(str(class_transition), '  ')}')
            print(f'[yellow]transition > 0[/]=\n{textwrap.indent(str((class_transition > 0).astype(int)), '  ')}')
            np.savetxt(f"output/isolate_{i}.txt", class_transition, fmt='%8s')

        output_states_prob = states_prob = solver.solve(len(class_states), class_transition, verbose=verbose)
        if verbose:
            print(f'{states_prob=!s}')

        # Compute time
        def delay_map(start: int, t: Transaction):
            available, offset = start
            delay = 0

            # Wait
            offset += t.wait
            if offset >= period:
                available = budget
                offset %= period

            # Length
            for l in t.lengths:
                # Wait till period over.
                if available <= 0:
                    delay += period - offset
                    available = budget
                    offset = 0

                # Do transaction.
                available -= l
                offset += l
                if offset >= period:
                    available = budget
                    offset %= period

            # Debug
            # print(f'{start} -> ({available}, {offset}): {delay}, {t}')
            return delay

        output_delay = delay = sum(
            delay_map(state, t) * tp * sp
            for t, tp in in_dist.dist.items()
            for state, sp in zip(class_states, states_prob)
        )
        if verbose:
            print(f'[magenta]E[R_i] (isolate)[/]={delay!s}')

        # # Compute delay
        # def period_map(start: int, t: Transaction):
        #     available, offset = start
        #     periods = 1
        #
        #     # Wait
        #     offset += t.wait
        #     if offset >= period:
        #         periods += offset // period
        #         available = budget
        #         offset %= period
        #
        #     # Length
        #     for l in t.lengths:
        #         # Wait till period over.]'.
        #         if available <= 0:
        #             periods += 1
        #             available = budget
        #             offset = 0
        #
        #         # Do transaction.
        #         available -= l
        #         offset += l
        #         if offset >= period:
        #             periods += offset // period
        #             available = budget
        #             offset %= period
        #
        #     # Debug
        #     # print(f'{start} -> ({available}, {offset}): {error}, {t}')
        #     return periods
        #
        # output_period = period = sum(
        #     period_map(state, t) * tp * sp
        #     for t, tp in in_dist.dist.items()
        #     for state, sp in zip(class_states, states_prob)
        # )
        # if verbose:
        #     print(f'[magenta]period (isolate)[/]={period!s}')

        # Compute delay
        def error_map(start: int, t: Transaction):
            available, offset = start

            # Wait
            offset += t.wait
            if offset >= period:
                yield int(available)
                yield from (budget for _ in range(int(offset // period - 1)))
                available = budget
                offset %= period

            # Length
            for l in t.lengths:
                # Wait till period over.
                if available <= 0:
                    yield int(available)
                    available = budget
                    offset = 0

                # Do transaction.
                available -= l
                offset += l
                if offset >= period:
                    yield int(available)
                    yield from (budget for _ in range(int(offset // period - 1)))
                    available = budget
                    offset %= period

        error = {}
        for t, tp in in_dist.dist.items():
            for state, sp in zip(class_states, states_prob):
                for rem in error_map(state, t):
                    error.setdefault(rem, 0)
                    error[rem] += tp * sp
        total = sum(error.values())
        error = {rem: p / total for rem, p in error.items()}
        output_error = error = sum(rem * p for rem, p in error.items())
        if verbose:
            print(f'[magenta]remaining  (isolate)[/]={error!s}')

        ### Compute transient delay
        if len(transients) > 0:
            mask = np.array([i in transients for i in range(len(transition))])
            tran_states = states[mask]
            tran_transition = transition[mask][:, mask]
            print(tran_transition)

            tran_hits = np.linalg.inv(np.eye(*tran_transition.shape) - tran_transition.astype(float))
            print(tran_hits[0, :])

            outgoing_delay = [
                sum(delay_map(state, t) * tp for t, tp in in_dist.dist.items())
                for state in tran_states
            ]
            print(outgoing_delay)

            transient_delay = sum(tran_hits[0, i] * outgoing_delay[i] for i in range(len(tran_states)))
            print(f'[dim]  truly transient delay: {transient_delay:.3f}')

            outgoing_delay = [
                sum(delay_map(state, t) * tp for t, tp in in_dist.dist.items())
                for state in class_states
            ]
            print(outgoing_delay)

            Pstep = np.eye(len(class_transition))
            steps = 0
            till_stationary_delay = 0
            while True:
                Pstep = Pstep @ class_transition
                steps += 1

                # Compute total variation distance
                tvd = np.max(0.5 * np.sum(np.abs(Pstep - states_prob), axis=1))
                print(f'[dim]->{tvd:.0e}, {Pstep[0, :]}')
                if tvd < 1e-6:
                    break

                till_stationary_delay += sum(Pstep[0, i] * outgoing_delay[i] for i in range(len(class_states)))

            print(Pstep)
            print(f'[dim]  mixing time: {steps - 1}')
            print(f'[dim]  till stationary delay: {till_stationary_delay:.6f}')

            transient_delay += till_stationary_delay
        else:
            transient_delay = 0

        if verbose:
            print(f'[magenta]E[R_T] (isolate)[/]={transient_delay:.6f}')

        # def make_data_and_ranges(filterO: int):
        #     mb, Mb = class_states[:, 0].min(), class_states[:, 0].max()
        #     mo, Mo = class_states[:, 1].min(), min(filterO, class_states[:, 1].max())
        #
        #     data = np.zeros((Mb - mb + 1, Mo - mo + 1))
        #     for s, p in zip(class_states, states_prob):
        #         b, o = s
        #         if o > filterO:
        #             continue
        #         data[b - mb, o - mo] = p
        #
        #     return data, mb, Mb, mo, Mo
        #
        # plot_dist_heatmap('model isolate states', *make_data_and_ranges(41))
    return output_error, output_delay, output_states, output_states_prob


def example():
    # Traffic configuration
    llow, lhigh = 1, 9
    wlow, whigh = 0, 4
    # Splitter configuration
    limit = 8
    # Isolate configuration
    K = 3
    budget = K * 1
    period = K * 4

    # Statistics part
    traffic_dist = Traffic.uniform(wlow, whigh, llow, lhigh)
    # print(traffic_dist)
    # print(f'traffic_bandwidth={traffic_dist.bandwidth()}')

    splitter_dist = traffic_dist.split(limit)
    # print(splitter_dist)
    # print(splitter_dist.equivalent_singular())
    # print(splitter_dist.bandwidth())
    # print(splitter_dist.equivalent_singular().bandwidth())
    print(splitter_dist)

    states, states_prob, buffer_delay = buffer_mc(splitter_dist, solver=Solver.SYMPY)
    # splitter_dist.plot('Ingresso')
    buffer_dist = splitter_dist.buffer(states, states_prob)
    buffer_dist.plot('Uscita')
    # print(f'buffer_bandwidth={buffer_dist.bandwidth()}')

    isolate_error, isolate_delay, isolate_states, isolate_states_prob = (
        isolate_mc(buffer_dist, budget, period, solver=Solver.HOPE)
    )
    print(f'[magenta]bandwidth[/]={buffer_dist.bandwidth(delay=isolate_delay)}')
    splitter_dist = traffic_dist.isolate(budget, period, isolate_states, isolate_states_prob)
    # splitter_dist.plot('Uscita')


def try_opt():
    # Traffic configuration
    llow, lhigh = 1, 9
    wlow, whigh = 0, 4
    traffic_dist = Traffic.uniform(wlow, whigh, llow, lhigh)

    ratio_target, ratio_prec = 0.50, 0.05
    PENALTY = 1e6  # Large value to discourage constraint violation

    @cache
    def compute(limit, budget, period):
        print(
            f'[cyan]compute[/]({limit!s}, {budget!s}/{period!s})'
        )
        splitter_dist = traffic_dist.split(limit)
        states, states_prob, buffer_delay = buffer_mc(splitter_dist, solver=Solver.SCIPY, verbose=False)
        buffer_dist = splitter_dist.buffer(states, states_prob)
        isolate_error, isolate_delay = isolate_mc(buffer_dist, budget, period, solver=Solver.SCIPY, verbose=False)

        bandwidth = traffic_dist.bandwidth(isolate_delay)
        error_precision = (budget - isolate_error) / budget
        R_T = buffer_delay
        R_i = isolate_delay
        print(
            f'[cyan]compute[/]({limit!s}, {budget!s}/{period!s})'
            f' -> {bandwidth:.4f}, {error_precision:.4f}, {R_T:.4f}, {R_i:.4f}'
        )
        return bandwidth, error_precision, R_T, R_i

    def objective(trial):
        # Suggest integer variables within bounds (replaces 'bounds' in scipy)
        limit = trial.suggest_int('limit', 4, 16)
        budget = trial.suggest_int('budget', 1, 64)
        period = trial.suggest_int('period', 1, 64)
        x = [limit, budget, period]

        # Compute all metrics once (assuming compute() is expensive)
        curr_ratio, curr_error_precision, R_T, R_i = compute(limit, budget, period)

        # Constraint handling through penalties
        penalty = 0.0

        # # 1. period >= budget constraint (from original ineq constraint)
        # if period < budget:
        #     penalty += BIG_PENALTY * (budget - period)

        # 2. Ratio matching constraint (original abs(ratio - compute(*x)[0]))
        # Assuming you want ratio_val close to some target ratio
        ratio_diff = abs(ratio_target - curr_ratio)
        if ratio_diff > ratio_prec:  # Add tolerance threshold
            print(f'Ratio penalty: {PENALTY * ratio_diff}')
            penalty += PENALTY * ratio_diff

        # 3. Error precision matching constraint
        # error_diff = abs(curr_error_precision - 1) - error_precision
        # if error_diff > error_precision:  # Add tolerance threshold
        #     print(f'Error penalty: {BIG_PENALTY * (error_diff - error_precision)}')
        #     penalty += BIG_PENALTY * (error_diff - error_precision)

        return R_i + penalty

    # Create and run the study
    import optuna
    study = optuna.create_study(
        direction='minimize',
        study_name=f'U({llow!r},{lhigh!r},{wlow!r},{whigh!r}),{ratio_target!r}#{ratio_prec!r}',
        storage="sqlite:///optuna_results.db",
        load_if_exists=True,  # Continue if study exists
    )

    study.optimize(objective, n_trials=15, n_jobs=3, show_progress_bar=True)
    print(study.best_params)

    def visual():
        import optuna.visualization as vis

        # 1. Optimization History Plot (shows progress over trials)
        fig_history = vis.plot_optimization_history(study)
        fig_history.show()

        # 2. Parameter Importance Plot (requires many trials)
        try:
            fig_param_importance = vis.plot_param_importances(study)
            fig_param_importance.show()
        except ValueError:
            print("Parameter importance requires completed trials with all parameters")

        # 3. Slice Plot (shows parameter values vs objective)
        fig_slice = vis.plot_slice(study, params=['limit', 'budget', 'period'])
        fig_slice.show()

        # 4. Parallel Coordinate Plot (shows parameter interactions)
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.show()

        # 5. Contour Plot (shows 2-parameter interactions)
        fig_contour = vis.plot_contour(study, params=['limit', 'budget', 'period'])
        fig_contour.show()

        # 6. Custom constraint violation visualization
        # Get all trials
        trials = study.trials_dataframe()

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trials['number'], y=trials['value'],
                                 mode='markers', name='Objective Value'))
        fig.update_layout(title='Objective Value with Constraint Satisfaction',
                          xaxis_title='Trial Number',
                          yaxis_title='Objective Value')
        fig.show()

        # 3D parameter space visualization
        fig_3d = go.Figure(data=go.Scatter3d(
            x=trials['params_limit'],
            y=trials['params_budget'],
            z=trials['params_period'],
            mode='markers',
            marker=dict(
                size=5,
                color=trials['value'],
                colorscale='Viridis',
                opacity=0.8
            )
        ))
        fig_3d.update_layout(scene=dict(
            xaxis_title='limit',
            yaxis_title='budget',
            zaxis_title='period'),
            title='Parameter Space Exploration')
        fig_3d.show()


if __name__ == '__main__':
    example()
