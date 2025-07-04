import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class RunResult:
    status: str
    time: float
    wtime: float
    mem: float
    bound: Optional[int]
    gotogen: Optional[float]
    gotoprocess: Optional[float]
    symex: Optional[float]
    slicing: Optional[float]
    solving: Optional[float]
    vcc: Optional[float]

    def is_unknown(self):
        return self.status in ["unknown", "TIMEOUT", "OUT OF MEMORY", "TIMEOUT (true)"]

    def is_correct(self, expected):
        fix = "false(unreach-call)" if expected == "false" else expected
        return self.status == fix


    def is_incorrect(self, expected):
        fix = "false(unreach-call)" if expected == "false" else expected
        return self.status != fix and not self.is_unknown()

@dataclass
class BenchmarkEntry:
    benchmark: str
    expected: str
    runs: List[RunResult] = field(default_factory=list)

def parse_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
def parse_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def parse_run_block(block, offset):
    return RunResult(
        status = block[0+offset],
        time = float(block[1+offset]),
        wtime = float(block[2+offset]),
        mem = float(block[3+offset]),
        bound = parse_int(block[4+offset]),
        gotogen = parse_float(block[5+offset]),
        gotoprocess = parse_float(block[6+offset]),
        symex = parse_float(block[7+offset]),
        slicing = parse_float(block[8+offset]),
        solving = parse_float(block[9+offset]),
        vcc = parse_float(block[10+offset])
    )

def parse_custom_csv(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()[3:]  # Skip first two lines
        reader = csv.reader(lines, delimiter='\t')
        for row in reader:
            entry = BenchmarkEntry(row[0], row[1], [parse_run_block(row,2), parse_run_block(row,13)])

            data.append(entry)
    return data


def print_status(blocks):
    correct_true = [0,0]
    correct_false = [0,0]
    incorrect_true = [0,0]
    incorrect_false = [0,0]

    for block in blocks:
        if block.expected == '':
            continue
        for index in [0,1]:
            if block.runs[index].is_correct(block.expected):
                if(block.runs[index].status == "true"):
                    correct_true[index] = correct_true[index]+1
                else:
                    correct_false[index] = correct_false[index]+1
            elif block.runs[index].is_incorrect(block.expected):
                if(block.runs[index].status == "true"):
                    incorrect_true[index] = incorrect_true[index]+1
                else:
                    incorrect_false[index] = incorrect_false[index]+1

    print(f'Status. Correct {correct_true} {correct_false}. Incorrect {incorrect_true} {incorrect_false}')


def draw(data1, data2, x_label, y_label, log=False):
    data1_sorted = sorted(data1)
    data2_sorted = sorted(data2)
    plt.plot(list(range(len(data1))), data1_sorted, label='Baseline', drawstyle='steps-post')
    plt.plot(list(range(len(data2))), data2_sorted, label='Transformed', drawstyle='steps-post')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if log:
        plt.yscale('log')
    plt.show()

def get_correct_times(blocks):
    data1 = []
    data2 = []

    for block in blocks:
        if block.expected == '':
            continue
        if block.runs[0].is_correct(block.expected):
            data1.append(block.runs[0].time)
        if block.runs[1].is_correct(block.expected):
            data2.append(block.runs[1].time)

    x = list(range(len(data1)))
    draw(data1, data2, "Benchmarks solved", "Total time (s)")

def get_correct_solving(blocks):
    data1 = []
    data2 = []

    for block in blocks:
        if block.expected == '':
            continue
        if block.runs[0].is_correct(block.expected):
            if block.runs[0].solving is None:
                data1.append(0)
            else:
                data1.append(block.runs[0].solving)
        if block.runs[1].is_correct(block.expected):
            if block.runs[1].solving is None:
                data2.append(0)
            else:
                data2.append(block.runs[1].solving)

    x = list(range(len(data1)))
    draw(data1, data2, "Benchmarks solved", "Total Solving time (s)")

def get_correct_memory_log(blocks):
    data1 = []
    data2 = []

    for block in blocks:
        if block.expected == '':
            continue
        if block.runs[0].is_correct(block.expected):
            data1.append(block.runs[0].mem)
        if block.runs[1].is_correct(block.expected):
            data2.append(block.runs[1].mem)

    x = list(range(len(data1)))
    draw(data1, data2, "Benchmarks solved", "Total memory (MB)", True)


def get_categories_unique(blocks):
    unique1 = []
    unique2 = []
    for block in blocks:
        if block.runs[0].status != block.runs[1].status:
            if block.runs[0].is_correct(block.expected):
                unique1.append((block.benchmark, block.runs[0]))
            elif block.runs[1].is_correct(block.expected):
                unique2.append((block.benchmark, block.runs[1]))

    print(f'Unique 1: {len(unique1)}')
    #print(unique1)

    print(f'Unique 2: {len(unique2)}')
    print(unique2)

if __name__ == "__main__":
    filename = "table.table.csv"
    raw_data = parse_custom_csv(filename)
    get_categories_unique(raw_data)
