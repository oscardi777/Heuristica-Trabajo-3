import os
import math
import time
import pandas as pd
import random


# ─────────────────────────────────────────────
# Archivos
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE_VND   = "resultados\\NWJSSP_OADG_NEH(VND).xlsx"



# ─────────────────────────────────────────────
# Parametros
# ─────────────────────────────────────────────
TIME_LIMIT_PER_BLOCK = 0.01
TIME_LIMIT_LS = 3600
MIXED_R = 0.2 # Porcentaje del vecindario a visitar para el segundo criterio
SEED = random.seed(42)

INSTANCES = [
    "ft06.txt",           "ft06r.txt",
    "ft10.txt",           "ft10r.txt",
    "ft20.txt",           "ft20r.txt",
    "tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
    "tai_j100_m10_1.txt",   "tai_j100_m10_1r.txt",
    "tai_j100_m100_1.txt",  "tai_j100_m100_1r.txt",
    "tai_j1000_m10_1.txt",  "tai_j1000_m10_1r.txt",
]


# ─────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────
class Operation:
    def __init__(self, machine, processing_time):
        self.machine = machine
        self.p = processing_time

class Job:
    def __init__(self, operations, release):
        self.operations = operations
        self.release = release

class Machine:
    def __init__(self, id: int):
        self.id = id
        self.intervals: list[tuple[int, int]] = []

    def add(self, b: int, e: int):
        self.intervals.append((b, e))

    def max_end_before(self, threshold: int):
        max_e = 0
        for b, e in self.intervals:
            if b < threshold:
                max_e = max(max_e, e)
        return max_e

def read_instance(file):
    with open(file) as f:
        n, m = map(int, f.readline().split())
        jobs = []
        for _ in range(n):
            data = list(map(int, f.readline().split()))
            operations = [Operation(data[2*i], data[2*i + 1]) for i in range(m)]
            jobs.append(Job(operations, release=data[-1]))
    return jobs, m

def precompute_offsets(jobs):
    return [[0] * len(job.operations) if len(job.operations) <= 1 else 
            [0] + [sum(op.p for op in job.operations[:u+1]) for u in range(len(job.operations)-1)]
            for job in jobs]

def compute_offsets(job):
    offsets = [0] * len(job.operations)
    total = 0
    for u, op in enumerate(job.operations[:-1]):
        total += op.p
        offsets[u + 1] = total
    return offsets

def find_start(job, machine_available, offsets):
    start = job.release
    for u, op in enumerate(job.operations):
        required = machine_available[op.machine] - offsets[u]
        if required > start:
            start = required
    return start

def schedule_job(job, machine_available, job_id, schedule):
    offsets = compute_offsets(job)
    start = find_start(job, machine_available, offsets)
    completion = 0
    for u, op in enumerate(job.operations):
        begin = start + offsets[u]
        finish = begin + op.p
        machine_available[op.machine] = finish
        if schedule is not None:
            schedule.append({"job": job_id, "machine": op.machine, "operation": u, "start": begin, "finish": finish})
        completion = finish
    return completion

def evaluate_sequence(sequence, jobs, m, save_schedule=False):
    machine_available = [0] * m
    total_flow = 0
    schedule = [] if save_schedule else None
    for j in sequence:
        Cj = schedule_job(jobs[j], machine_available, j, schedule)
        total_flow += Cj
    return (total_flow, schedule) if save_schedule else total_flow

def evaluate_insertion(sequence, j, pos, jobs, m):
    machine_available = [0] * m
    total_flow = 0
    for idx in range(pos):
        total_flow += schedule_job(jobs[sequence[idx]], machine_available, sequence[idx], None)
    total_flow += schedule_job(jobs[j], machine_available, j, None)
    for idx in range(pos, len(sequence)):
        total_flow += schedule_job(jobs[sequence[idx]], machine_available, sequence[idx], None)
    return total_flow

def find_start_preciso(job, machines, offsets):
    start = job.release
    while True:
        max_candidate = start
        feasible = True
        for u, op in enumerate(job.operations):
            b_op = start + offsets[u]
            e_op = b_op + op.p
            max_ek = machines[op.machine].max_end_before(e_op)
            if max_ek > b_op:
                feasible = False
                candidate = max_ek - offsets[u]
                max_candidate = max(max_candidate, candidate)
        if feasible:
            return start
        start = max_candidate

def schedule_job_preciso(job, machines, job_id, schedule, offsets):
    start = find_start_preciso(job, machines, offsets)
    completion = 0
    for u, op in enumerate(job.operations):
        begin = start + offsets[u]
        finish = begin + op.p
        machines[op.machine].add(begin, finish)
        if schedule is not None:
            schedule.append({"job": job_id, "machine": machines[op.machine].id, "operation": u, "start": begin, "finish": finish})
        completion = finish
    return completion

def evaluate_sequence_preciso(sequence, jobs, m, offsets_list, save_schedule=False):
    machines = [Machine(i) for i in range(m)]
    total_flow = 0
    schedule = [] if save_schedule else None
    for j in sequence:
        total_flow += schedule_job_preciso(jobs[j], machines, j, schedule, offsets_list[j])
    return (total_flow, schedule) if save_schedule else total_flow

def find_best_insertion(sequence, j, jobs, m, block_size, time_limit):
    n_pos = len(sequence) + 1
    best_pos = 0
    best_value = float("inf")
    pos = 0
    while pos < n_pos:
        end_block = min(pos + block_size, n_pos)
        t_bloque = time.time()
        for p in range(pos, end_block):
            if time.time() - t_bloque > time_limit:
                break
            value = evaluate_insertion(sequence, j, p, jobs, m)
            if value < best_value:
                best_value = value
                best_pos = p
        pos = end_block
    return best_pos, best_value

def construct_solution(jobs, m):
    n = len(jobs)
    block_size = max(10, int(math.sqrt(n)))
    order = sorted(range(n), key=lambda j: jobs[j].release + sum(op.p for op in jobs[j].operations), reverse=True)
    sequence = []
    for j in order:
        best_pos, _ = find_best_insertion(sequence, j, jobs, m, block_size, TIME_LIMIT_PER_BLOCK)
        sequence.insert(best_pos, j)
    return sequence

def write_results_to_excel(results, output_file):
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    writer_kwargs = dict(engine="openpyxl", mode="a", if_sheet_exists="replace") if os.path.exists(output_file) else dict(engine="openpyxl", mode="w")
    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:
        for sheet_name, (total_flow, compute_time_ms, job_start_times) in results.items():
            df = pd.DataFrame([[total_flow, compute_time_ms], job_start_times])
            df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
    print(f"Resultados guardados en: {output_file}")

# ─────────────────────────────────────────────
# Generador de vecindarios
# ─────────────────────────────────────────────
def generate_insertion_down_neighbors(sequence):
    neighbors = []
    n = len(sequence)
    for i in range(n):
        for j in range(i + 1, n):
            new_seq = sequence[:]
            job = new_seq.pop(i)
            new_seq.insert(j, job)
            neighbors.append(new_seq)
    return neighbors

def generate_swap_neighbors(sequence):
    neighbors = []
    n = len(sequence)
    for i in range(n):
        for j in range(i + 1, n):             # solo i < j → sin duplicados
            new_seq = sequence[:]
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            neighbors.append(new_seq)
    return neighbors

def generate_insertion_up_neighbors(sequence):
    neighbors = []
    n = len(sequence)
    for i in range(n):
        for j in range(i):
            new_seq = sequence[:]
            job = new_seq.pop(i)
            new_seq.insert(j, job)
            neighbors.append(new_seq)
    return neighbors

# ─────────────────────────────────────────────
# FIRST IMPROVEMENT
# ─────────────────────────────────────────────
def local_search_first_improvement(initial_sequence, jobs, m, offsets_list, start_time, k, current_z, NEIGHBORS_GENERATOR):
    B = list(initial_sequence)

    fin = False
    while not fin and (time.time() - start_time < 3600):
        fin = True
        neighbors = NEIGHBORS_GENERATOR[k](B)
        for P in neighbors:
            if time.time() - start_time >= 3600:
                break
            z = evaluate_sequence_preciso(P, jobs, m, offsets_list)
            if z < current_z:
                B = P
                current_z = z
                fin = False
                break

    return B, current_z, time.time()

# ─────────────────────────────────────────────
# VND
# ─────────────────────────────────────────────
def vnd(initial_sequence, jobs, m, offsets_list, start_time):
    B = list(initial_sequence)
    current_z = evaluate_sequence_preciso(B, jobs, m, offsets_list)

    NEIGHBORS_GENERATOR =[generate_insertion_down_neighbors, generate_swap_neighbors, generate_insertion_up_neighbors]
    k = 0
    while ((time.time() - start_time < 3600) and k < len(NEIGHBORS_GENERATOR)):
        candidate, z_candidate, _ = local_search_first_improvement(B, jobs, m, offsets_list, start_time, k, current_z, NEIGHBORS_GENERATOR)
        if z_candidate < current_z:
            k=0
            B = candidate
            current_z = z_candidate
        else:
            k += 1

    return B, current_z, time.time()



# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    for inst in INSTANCES:
        filepath = os.path.join(INSTANCES_DIR, inst)
        if not os.path.exists(filepath):
            print(f"[SKIP] {inst} — archivo no encontrado")
            continue

        jobs, m = read_instance(filepath)
        n = len(jobs)
        sheet_name = inst.replace(".txt", "")

        # Timer inicia ANTES de NEH
        t0 = time.time()

        # 1. Solución inicial mediante NEH
        sequence = construct_solution(jobs, m)
        print(f"\n[LOADING] | Procesando instancia | {inst} ")

        offsets_list = precompute_offsets(jobs)

        # 2. VND
        seq_vnd, z_vnd, time_finish_VND = vnd(sequence, jobs, m, offsets_list, t0)

        compute_time_vnd = round((time_finish_VND - t0) * 1000)

        # Schedule para Excel
        _, schedule = evaluate_sequence_preciso(seq_vnd, jobs, m, offsets_list, save_schedule=True)
        job_start_times = [None] * n
        for op in schedule:
            if op["operation"] == 0:
                job_start_times[op["job"]] = op["start"]

        single_results = {sheet_name: (z_vnd, compute_time_vnd, job_start_times)}
        write_results_to_excel(single_results, OUTPUT_FILE_VND)

if __name__ == "__main__":
    main()