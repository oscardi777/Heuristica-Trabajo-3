import os
import math
import random
import time
import pandas as pd


# ─────────────────────────────────────────────
# Archivos
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE = "resultados\\NWJSSP_OADG_NEH(MS+ELS+PROBABILISTICO).xlsx"


# ─────────────────────────────────────────────
# Parametros
# ─────────────────────────────────────────────
ALPHA = 0.20
N_SOL = 3
N_ITER = 10
N_CANDIDATES = 5
TIME_LIMIT_TOTAL = 3600
TIME_LIMIT_PER_BLOCK = 0.01

# Probabilidad de aceptar una solucion peor durante el ELS (0.0 = nunca, 1.0 = siempre)
# Solo aplica en la fase ELS, NO en la construccion inicial ni en la busqueda local pura.
PROB_ACCEPT_WORSE = 0.10

random.seed(42)

INSTANCES = [
    #"ft06.txt",           "ft06r.txt",
    #"ft10.txt",           "ft10r.txt",
    #"ft20.txt",           "ft20r.txt",
    "tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
    #"tai_j100_m10_1.txt",   "tai_j100_m10_1r.txt",
    #"tai_j100_m100_1.txt",  "tai_j100_m100_1r.txt",
    #"tai_j1000_m10_1.txt",  "tai_j1000_m10_1r.txt",
    #"tai_j1000_m100_1.txt", "tai_j1000_m100_1r.txt",
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
    """Registra los intervalos ocupados de una maquina para el scheduling preciso."""

    def __init__(self, machine_id: int):
        self.id = machine_id
        self.intervals = []

    def add_interval(self, start: int, end: int):
        self.intervals.append((start, end))

    def latest_end_before(self, threshold: int):
        return max((e for b, e in self.intervals if b < threshold), default=0)


# ─────────────────────────────────────────────
# Lectura de instancias
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


# ─────────────────────────────────────────────
# Computo de offsets
# ─────────────────────────────────────────────
def compute_offsets(job: Job):
    """Calcula el tiempo acumulado de procesamiento antes de cada operacion.
    Permite calcular el inicio de cada operacion dado un unico tiempo de arranque."""
    offsets, cumulative = [0] * len(job.operations), 0
    for u, op in enumerate(job.operations[:-1]):
        cumulative += op.p
        offsets[u + 1] = cumulative
    return offsets


def precompute_all_offsets(jobs: list):
    return [compute_offsets(job) for job in jobs]


# ─────────────────────────────────────────────
# Scheduling aproximado (construccion meta)
# ─────────────────────────────────────────────
def find_start_approximate(job: Job, machine_available: list, offsets: list):
    """Calcula el menor tiempo de inicio factible ignorando solapamientos entre
    operaciones del mismo job en maquinas compartidas (aproximacion para la construccion)."""
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
            schedule.append({
                "job": job_id, "machine": op.machine,
                "operation": u, "start": begin, "finish": finish,
            })
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


# ─────────────────────────────────────────────
# Scheduling preciso (busqueda local)
# ─────────────────────────────────────────────
def find_start_precise(job: Job, machines: list, offsets: list):
    """Calcula el menor tiempo de inicio factible respetando los intervalos
    reales ocupados en cada maquina (sin solapamientos)."""
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
            schedule.append({
                "job": job_id, "machine": machines[op.machine].id,
                "operation": u, "start": begin, "finish": finish,
            })
        completion = finish
    return completion


def evaluate_sequence_precise(sequence: list, jobs: list, m: int, offsets_list: list, save_schedule: bool = False):
    machines = [MachineTracker(i) for i in range(m)]
    schedule = [] if save_schedule else None
    total_flow = sum(
        schedule_job_precise(jobs[j], machines, j, offsets_list[j], schedule)
        for j in sequence
    )
    return (total_flow, schedule) if save_schedule else total_flow


# ─────────────────────────────────────────────
# Construccion de solucion meta
# ─────────────────────────────────────────────
def build_rcl(pending: list, jobs: list, alpha: float):
    """Construye la Lista Restringida de Candidatos basada en valor.
    alpha=0 es greedy puro (solo el mejor), alpha=1 es aleatorio puro (todos).

    BUG CORREGIDO: el threshold original usaba (w_max - alpha*(w_max-w_min)),
    que excluia a los jobs mas ligeros siendo que NEH los quiere primero.
    Ahora se incluyen todos los jobs con peso >= w_min + (1-alpha)*(w_max-w_min),
    lo que con alpha=0 deja solo el mejor y con alpha=1 incluye todos.
    """
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
    """Una iteracion de construccion meta integrada en el marco NEH.
    La iteracion con alpha=0 equivale al NEH greedy clasico."""
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
# Generadores de vecindarios
# ─────────────────────────────────────────────
def insertion_forward_neighbors(sequence: list):
    """Mueve cada job hacia adelante en la secuencia (i -> j, j > i)."""
    n = len(sequence)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = sequence[:]
            job = neighbor.pop(i)
            neighbor.insert(j, job)
            yield neighbor


def insertion_backward_neighbors(sequence: list):
    """Mueve cada job hacia atras en la secuencia (i -> j, j < i)."""
    n = len(sequence)
    for i in range(n):
        for j in range(i):
            neighbor = sequence[:]
            job = neighbor.pop(i)
            neighbor.insert(j, job)
            yield neighbor


def swap_neighbors(sequence: list):
    """Intercambia cada par de jobs (i, j) con i < j."""
    n = len(sequence)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = sequence[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            yield neighbor


NEIGHBORHOOD_GENERATOR = insertion_backward_neighbors


# ─────────────────────────────────────────────
# Busqueda local (First Improvement)
# ─────────────────────────────────────────────
def first_improvement_local_search(
    sequence: list,
    jobs: list,
    m: int,
    offsets_list: list,
    start_time: float,
    current_value: int,
    accept_worse: bool = False,
    prob_accept_worse: float = PROB_ACCEPT_WORSE,
):
    """Busqueda local First Improvement.

    accept_worse=False (default): comportamiento clasico, solo acepta mejoras.
    accept_worse=True           : acepta con probabilidad `prob_accept_worse`
                                  una solucion vecina que NO mejore la actual.
                                  Solo se usa durante el ELS para escapar de
                                  optimos locales sin necesidad de recocido simulado.
    """
    best_sequence = list(sequence)
    improved = True
    while improved and (time.time() - start_time < TIME_LIMIT_TOTAL):
        improved = False
        for neighbor in NEIGHBORHOOD_GENERATOR(best_sequence):
            if time.time() - start_time >= TIME_LIMIT_TOTAL:
                break
            value = evaluate_sequence_precise(neighbor, jobs, m, offsets_list)
            if value < current_value:
                # Siempre aceptar mejora
                best_sequence = neighbor
                current_value = value
                improved = True
                break
            elif accept_worse and value > current_value:
                # Aceptar solucion peor con probabilidad baja
                if random.random() < prob_accept_worse:
                    best_sequence = neighbor
                    current_value = value
                    improved = True  # forzar otra ronda desde este nuevo punto
                    break
    return best_sequence, current_value


# ─────────────────────────────────────────────
# Perturbacion
# ─────────────────────────────────────────────
def perturbation(sequence: list, jobs: list, m: int, offsets_list: list):
    """Realiza una perturbacion aleatoria de acuerdo al movimiento del vecindario.

    BUG CORREGIDO: el 'continue' cuando i==j podia dejar perturbaciones sin
    aplicar. Ahora se reintenta hasta obtener un par (i, j) valido.
    """
    new_seq = sequence[:]
    n = len(new_seq)
    npert = 3

    for _ in range(npert):
        # Reintentar hasta obtener indices distintos y validos para el movimiento
        max_attempts = 20
        for _ in range(max_attempts):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)

            if i == j:
                continue

            if NEIGHBORHOOD_GENERATOR == insertion_forward_neighbors:
                if i < j:
                    job = new_seq.pop(i)
                    new_seq.insert(j, job)
                    break
            elif NEIGHBORHOOD_GENERATOR == insertion_backward_neighbors:
                if i > j:
                    job = new_seq.pop(i)
                    new_seq.insert(j, job)
                    break
            else:  # swap
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                break

    new_value = evaluate_sequence_precise(new_seq, jobs, m, offsets_list)
    return new_seq, new_value


# ─────────────────────────────────────────────
# meta: MultiStart + ELS + Probabilistico Simple
# ─────────────────────────────────────────────
def meta(jobs: list, m: int, offsets_list: list, start_time: float):
    """
    MultiStart + ELS + Probabilistico Simple.

    Estructura:
      Para cada solucion inicial (MultiStart):
        1. Construccion meta (GRASP-NEH)
        2. Busqueda local DETERMINISTA (accept_worse=False)  <- sin aceptar peores
        3. ELS sobre la mejor local:
             - Perturbacion
             - Busqueda local PROBABILISTICA (accept_worse=True) <- acepta peores con prob baja
             - Criterio de aceptacion de la base del ELS: solo si mejora
             - Actualizacion del optimo global: solo si mejora
    """
    n = len(jobs)
    block_size = max(10, int(math.sqrt(n)))

    best_sequence = None
    best_value = float("inf")

    alpha = ALPHA
    for h in range(N_SOL):
        if time.time() - start_time >= TIME_LIMIT_TOTAL:
            break

        # ── 1. Construccion + Busqueda local inicial (SIN aceptar peores) ──
        s = build_meta_solution(jobs, m, alpha, block_size)
        s_val = evaluate_sequence_precise(s, jobs, m, offsets_list)
        s, f_s = first_improvement_local_search(
            s, jobs, m, offsets_list, start_time,
            current_value=s_val,
            accept_worse=False,          # <-- determinista
        )

        # Actualizar mejor solucion global
        if f_s < best_value:
            best_value = f_s
            best_sequence = s[:]

        # Base del ELS para esta solucion de inicio
        current_s = s[:]
        current_f = f_s

        # ── 2. ELS con busqueda local probabilistica ──
        for it in range(N_ITER):
            if time.time() - start_time >= TIME_LIMIT_TOTAL:
                break

            best_f_candidate = float("inf")
            best_candidate = None

            for c in range(N_CANDIDATES):
                if time.time() - start_time >= TIME_LIMIT_TOTAL:
                    break

                # Perturbacion desde la solucion base del ELS
                s_pert, f_pert = perturbation(current_s, jobs, m, offsets_list)

                s_new, f_new = first_improvement_local_search(
                    s_pert, jobs, m, offsets_list, start_time,
                    current_value=f_pert,
                    accept_worse=True,           # <-- probabilistico
                    prob_accept_worse=PROB_ACCEPT_WORSE,
                )

                if f_new < best_f_candidate:
                    best_f_candidate = f_new
                    best_candidate = s_new[:]

            # Actualizar mejor solucion global
            if best_f_candidate < best_value:
                best_value = best_f_candidate
                best_sequence = best_candidate[:]

            # Actualizar base del ELS solo si hay mejora (criterio conservador)
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
    writer_kwargs = (
        dict(engine="openpyxl", mode="a", if_sheet_exists="replace")
        if os.path.exists(output_file)
        else dict(engine="openpyxl", mode="w")
    )
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