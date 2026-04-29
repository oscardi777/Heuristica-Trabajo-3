import os
import math
import random
import time
import pandas as pd

# ─────────────────────────────────────────────
# Archivos y Parámetros
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE = "resultados\\Exp_alpha0.5_NWJSSP_OADG_NEH(MS+ELS+SA).xlsx"

ALPHA = 0.20
N_SOL = 3
N_ITER_ELS = 10
N_CANDIDATES = 5
N_PERTURBACIONES = 3
TIME_LIMIT_TOTAL = 3600
TIME_LIMIT_PER_BLOCK = 0.01

# Parámetros Recocido Simulado
T0 = 100.0          # Temperatura inicial
TF = 0.01           # Temperatura final
ALPHA_COOLING = 0.5  # Factor de enfriamiento
L = 20              # Iteraciones por nivel de temperatura

random.seed(42)

INSTANCES = [
    #"ft06.txt",           "ft06r.txt",
    #"ft10.txt",           "ft10r.txt",
    #"ft20.txt",           "ft20r.txt",
    "tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
    "tai_j100_m10_1.txt",   "tai_j100_m10_1r.txt",
    "tai_j100_m100_1.txt",  "tai_j100_m100_1r.txt",
    "tai_j1000_m10_1.txt",  "tai_j1000_m10_1r.txt",
]

# ─────────────────────────────────────────────
# Estructuras de datos 
# ─────────────────────────────────────────────
class Operation:
    def __init__(self, machine: int, processing_time: int):
        self.machine = machine
        self.p = processing_time

class Job:
    def __init__(self, operations: list, release: int):
        self.operations = operations
        self.release = release

class MachineTracker:
    def __init__(self, machine_id: int):
        self.id = machine_id
        self.intervals = []

    def add_interval(self, start: int, end: int):
        self.intervals.append((start, end))

    def latest_end_before(self, threshold: int):
        return max((e for b, e in self.intervals if b < threshold), default=0)

# ─────────────────────────────────────────────
# Lectura y offsets 
# ─────────────────────────────────────────────
def read_instance(filepath: str):
    with open(filepath) as f:
        n, m = map(int, f.readline().split())
        jobs = []
        for _ in range(n):
            data = list(map(int, f.readline().split()))
            operations = [Operation(data[2*i], data[2*i + 1]) for i in range(m)]
            release = data[-1]
            jobs.append(Job(operations, release))
    return jobs, m

def compute_offsets(job: Job):
    offsets, cumulative = [0] * len(job.operations), 0
    for u, op in enumerate(job.operations[:-1]):
        cumulative += op.p
        offsets[u + 1] = cumulative
    return offsets

def precompute_all_offsets(jobs: list):
    return [compute_offsets(job) for job in jobs]

# ─────────────────────────────────────────────
# Scheduling aproximado y preciso 
# ─────────────────────────────────────────────
def find_start_approximate(job: Job, machine_available: list, offsets: list):
    start = job.release
    for u, op in enumerate(job.operations):
        start = max(start, machine_available[op.machine] - offsets[u])
    return start

def schedule_job_approximate(job: Job, machine_available: list, job_id: int, schedule):
    offsets = compute_offsets(job)
    start = find_start_approximate(job, machine_available, offsets)
    completion = 0
    for u, op in enumerate(job.operations):
        begin = start + offsets[u]
        finish = begin + op.p
        machine_available[op.machine] = finish
        if schedule is not None:
            schedule.append({"job": job_id, "machine": op.machine, "operation": u, "start": begin, "finish": finish})
        completion = finish
    return completion

def evaluate_insertion_approximate(sequence: list, job_to_insert: int, insert_pos: int, jobs: list, m: int):
    machine_available = [0] * m
    total_flow = 0
    for idx in range(insert_pos):
        total_flow += schedule_job_approximate(jobs[sequence[idx]], machine_available, sequence[idx], None)
    total_flow += schedule_job_approximate(jobs[job_to_insert], machine_available, job_to_insert, None)
    for idx in range(insert_pos, len(sequence)):
        total_flow += schedule_job_approximate(jobs[sequence[idx]], machine_available, sequence[idx], None)
    return total_flow

def find_start_precise(job: Job, machines: list, offsets: list):
    start = job.release
    while True:
        best_candidate = start
        feasible = True
        for u, op in enumerate(job.operations):
            op_start = start + offsets[u]
            op_end = op_start + op.p
            latest_busy = machines[op.machine].latest_end_before(op_end)
            if latest_busy > op_start:
                feasible = False
                candidate = latest_busy - offsets[u]
                best_candidate = max(best_candidate, candidate)
        if feasible:
            return start
        start = best_candidate

def schedule_job_precise(job: Job, machines: list, job_id: int, offsets: list, schedule):
    start = find_start_precise(job, machines, offsets)
    completion = 0
    for u, op in enumerate(job.operations):
        begin = start + offsets[u]
        finish = begin + op.p
        machines[op.machine].add_interval(begin, finish)
        if schedule is not None:
            schedule.append({"job": job_id, "machine": machines[op.machine].id, "operation": u, "start": begin, "finish": finish})
        completion = finish
    return completion

def evaluate_sequence_precise(sequence: list, jobs: list, m: int, offsets_list: list, save_schedule: bool = False):
    machines = [MachineTracker(i) for i in range(m)]
    schedule = [] if save_schedule else None
    total_flow = sum(schedule_job_precise(jobs[j], machines, j, offsets_list[j], schedule) for j in sequence)
    return (total_flow, schedule) if save_schedule else total_flow

# ─────────────────────────────────────────────
# Construcción meta 
# ─────────────────────────────────────────────
def build_rcl(pending: list, jobs: list, alpha: float):
    weights = {j: jobs[j].release + sum(op.p for op in jobs[j].operations) for j in pending}
    w_max = max(weights.values())
    w_min = min(weights.values())
    threshold = w_min + (1.0 - alpha) * (w_max - w_min)
    return [j for j in pending if weights[j] >= threshold]

def find_best_insertion_position(sequence: list, job_to_insert: int, jobs: list, m: int, block_size: int):
    n_positions = len(sequence) + 1
    best_pos = 0
    best_value = float("inf")
    pos = 0
    while pos < n_positions:
        block_end = min(pos + block_size, n_positions)
        block_start = time.time()
        for p in range(pos, block_end):
            if time.time() - block_start > TIME_LIMIT_PER_BLOCK:
                break
            value = evaluate_insertion_approximate(sequence, job_to_insert, p, jobs, m)
            if value < best_value:
                best_value = value
                best_pos = p
        pos = block_end
    return best_pos, best_value

def build_meta_solution(jobs: list, m: int, alpha: float, block_size: int):
    pending = list(range(len(jobs)))
    sequence = []
    while pending:
        rcl = build_rcl(pending, jobs, alpha)
        j = random.choice(rcl)
        pending.remove(j)
        best_pos, _ = find_best_insertion_position(sequence, j, jobs, m, block_size)
        sequence.insert(best_pos, j)
    return sequence

# ─────────────────────────────────────────────
# Vecindario 
# ─────────────────────────────────────────────
def insertion_backward_neighbors(sequence: list):
    n = len(sequence)
    for i in range(n):
        for j in range(i):
            neighbor = sequence[:]
            job = neighbor.pop(i)
            neighbor.insert(j, job)
            yield neighbor

NEIGHBORHOOD_GENERATOR = insertion_backward_neighbors

# ─────────────────────────────────────────────
# Recocido Simulado como Búsqueda Local
# ─────────────────────────────────────────────
def simulated_annealing_local_search(sequence, jobs, m, offsets_list, start_time, current_value, T0_sa=None):
    if T0_sa is None:
        T0_sa = T0
    
    s = list(sequence)
    f_s = current_value
    best_s = list(s)
    best_f = f_s

    T = T0_sa
    while T > TF and (time.time() - start_time < TIME_LIMIT_TOTAL):
        for _ in range(L):
            if time.time() - start_time >= TIME_LIMIT_TOTAL:
                break
                
            # Generar vecino
            for neighbor in NEIGHBORHOOD_GENERATOR(s):
                if time.time() - start_time >= TIME_LIMIT_TOTAL:
                    break
                f_neighbor = evaluate_sequence_precise(neighbor, jobs, m, offsets_list)
                
                delta = f_neighbor - f_s
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    s = neighbor
                    f_s = f_neighbor
                    
                    if f_s < best_f:
                        best_f = f_s
                        best_s = list(s)
                    break  # First Improvement style dentro de SA
        
        T *= ALPHA_COOLING  # Enfriamiento geométrico
    
    return best_s, best_f

# ─────────────────────────────────────────────
# Perturbación 
# ─────────────────────────────────────────────
def perturbation(sequence: list, jobs: list, m: int, offsets_list: list):
    """Perturbación fuerte: mezcla inserciones grandes + swaps múltiples."""
    new_seq = sequence[:]
    n = len(new_seq)

    for _ in range(N_PERTURBACIONES):
        if random.random() < 0.7:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            while abs(i - j) < 2:
                j = random.randint(0, n-1)
            job = new_seq.pop(i)
            new_seq.insert(j, job)
        else:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            while i == j:
                j = random.randint(0, n-1)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

    new_value = evaluate_sequence_precise(new_seq, jobs, m, offsets_list)
    return new_seq, new_value

# ─────────────────────────────────────────────
# Metaheurística: MultiStart + ELS + Recocido Simulado
# ─────────────────────────────────────────────
def meta(jobs: list, m: int, offsets_list: list, start_time: float):
    n = len(jobs)
    block_size = max(10, int(math.sqrt(n)))

    best_sequence = None
    best_value = float("inf")

    for h in range(N_SOL):
        if time.time() - start_time >= TIME_LIMIT_TOTAL:
            break

        # 1. Construcción GRASP-NEH + Búsqueda Local determinista
        s = build_meta_solution(jobs, m, ALPHA, block_size)
        s_val = evaluate_sequence_precise(s, jobs, m, offsets_list)
        
        s, f_s = simulated_annealing_local_search(s, jobs, m, offsets_list, start_time, s_val, T0_sa=10.0)  # Temperatura baja para fase inicial

        if f_s < best_value:
            best_value = f_s
            best_sequence = s[:]

        # 2. ELS con Recocido Simulado
        current_s = s[:]
        current_f = f_s

        for it in range(N_ITER_ELS):
            if time.time() - start_time >= TIME_LIMIT_TOTAL:
                break

            best_f_candidate = float("inf")
            best_candidate = None

            for c in range(N_CANDIDATES):
                if time.time() - start_time >= TIME_LIMIT_TOTAL:
                    break

                s_pert, f_pert = perturbation(current_s, jobs, m, offsets_list)

                # Aquí se aplica Recocido Simulado completo sobre la perturbación
                s_new, f_new = simulated_annealing_local_search(
                    s_pert, jobs, m, offsets_list, start_time, f_pert, T0_sa=T0
                )

                if f_new < best_f_candidate:
                    best_f_candidate = f_new
                    best_candidate = s_new[:]

            if best_f_candidate < best_value:
                best_value = best_f_candidate
                best_sequence = best_candidate[:]

            if best_f_candidate < current_f:
                current_s = best_candidate[:]
                current_f = best_f_candidate

    return best_sequence, best_value

# ─────────────────────────────────────────────
# Guardado de resultados
# ─────────────────────────────────────────────
def extract_job_start_times(schedule: list, n: int):
    job_start_times = [None] * n
    for op in schedule:
        if op["operation"] == 0:
            job_start_times[op["job"]] = op["start"]
    return job_start_times

def write_results_to_excel(results: dict, output_file: str):
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    writer_kwargs = dict(engine="openpyxl", mode="a", if_sheet_exists="replace") if os.path.exists(output_file) else dict(engine="openpyxl", mode="w")
    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:
        for sheet_name, (total_flow, elapsed_ms, job_start_times) in results.items():
            df = pd.DataFrame([[total_flow, elapsed_ms], job_start_times])
            df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
    print(f"Resultados guardados en: {output_file}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    for instance_file in INSTANCES:
        filepath = os.path.join(INSTANCES_DIR, instance_file)
        if not os.path.exists(filepath):
            print(f"[SKIP] {instance_file} — archivo no encontrado")
            continue

        jobs, m = read_instance(filepath)
        n = len(jobs)
        sheet_name = instance_file.replace(".txt", "")

        print(f"\n[LOADING] | Procesando instancia | {instance_file}")
        t0 = time.time()

        offsets_list = precompute_all_offsets(jobs)

        best_sequence, best_value = meta(jobs, m, offsets_list, t0)

        elapsed_ms = round((time.time() - t0) * 1000)

        _, schedule = evaluate_sequence_precise(best_sequence, jobs, m, offsets_list, save_schedule=True)
        job_start_times = extract_job_start_times(schedule, n)

        write_results_to_excel(
            {sheet_name: (best_value, elapsed_ms, job_start_times)},
            OUTPUT_FILE,
        )

if __name__ == "__main__":
    main()