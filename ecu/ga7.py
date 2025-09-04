import random
import copy
from collections import defaultdict, deque
import math
import itertools
import time
import tracemalloc
import concurrent.futures

alpha = 0.7
beta = 0.3

# DELAY_MATRIX=[
#     [0, 2, 1, 1],
#     [2, 0, 1, 2],
#     [1, 1, 0, 2],
#     [1, 2, 2, 0]
# ]
DELAY_MATRIX=[
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
def find_minimal_ecu_covers(subtask, ecus):
    required=subtask['resources']
    ecu_indices=list(range(len(ecus)))
    for r in range(1, len(ecus)+1):
        for combo in itertools.combinations(ecu_indices, r):
            combined=set()
            for idx in combo:
                combined |=ecus[idx]['resources']
            if required.issubset(combined):
                # Minimality check
                minimal=True
                for i in range(r):
                    test_combo=combo[:i]+combo[i+1:]
                    test_combined=set()
                    for idx in test_combo:
                        test_combined |=ecus[idx]['resources']
                    if required.issubset(test_combined):
                        minimal=False
                        break
                if minimal:
                    yield combo

def to_assignment_list(assignment):
    if isinstance(assignment, list):
        return assignment
    elif isinstance(assignment, tuple):
        return [assignment]
    elif isinstance(assignment, int):
        return [(assignment, 0)]
    else:
        raise ValueError("Invalid assignment type")


                    
def subtask_exec_time(subtask):
    return math.ceil(subtask['x']/subtask['n']) * subtask['t']
def critical_path_length(subtasks, edges):
    n = len(subtasks)
    adj = [[] for _ in range(n)]
    indegree = [0]*n
    for u,v in edges:
        adj[u-1].append(v-1)
        indegree[v-1] += 1
    longest_path = [0]*n

    # Topological order
    q = deque([i for i in range(n) if indegree[i]==0])

    while q:
        u = q.popleft()
        u_time = longest_path[u] + subtask_exec_time(subtasks[u])
        for v in adj[u]:
            if longest_path[v] < u_time:
                longest_path[v] = u_time
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    # The finish time for each node is its start time (longest_path) + its exec time
    finish_times = [longest_path[i] + subtask_exec_time(subtasks[i]) for i in range(n)]
    return max(finish_times) if finish_times else 0



def compute_critical_path(dag):
    subtasks = dag['subtasks']
    edges = dag['edges']
    subtask_count = len(subtasks)
    adj = [[] for _ in range(subtask_count)]
    for u, v in edges:
        adj[u-1].append(v-1)
    dp = [0] * subtask_count
    order = topological_order(subtask_count, edges)
    for u in order:
        x, n, t = subtasks[u]['x'], subtasks[u]['n'], subtasks[u]['t']
        exec_time = math.ceil(x/n) * t
        for v in adj[u]:
            dp[v] = max(dp[v], dp[u]+ exec_time)
    for u in range(subtask_count):
        x, n, t = subtasks[u]['x'], subtasks[u]['n'], subtasks[u]['t']
        exec_time = math.ceil(x/n) * t
        dp[u]+= exec_time
    return max(dp)

def compute_total_workload(dag):
    subtasks = dag['subtasks']
    total = 0
    for subtask in subtasks:
        x, n, t = subtask['x'], subtask['n'], subtask['t']
        total+= math.ceil(x/n) * t
    return total

def merge_dags(dag_list):
    merged_subtasks=[]
    merged_edges=[]
    offset=0 

    for dag in dag_list:
        merged_subtasks.extend(dag['subtasks'])
        for u,v in dag['edges']:
            merged_edges.append((u+offset, v+offset))
        offset+=len(dag['subtasks'])
    return merged_subtasks, merged_edges

def compute_theoretical_bounds(dag_list):
    # Merge all DAGs first
    subtasks, edges = merge_dags(dag_list)
    
    # Cref_min: critical path of the merged DAG (not just max of individual DAGs)
    Cref_min = critical_path_length(subtasks, edges)
    
    # Cref_max: total workload of all subtasks
    Cref_max = sum(subtask_exec_time(st) for st in subtasks)
    
    Dref_min = 0
    Dref_max = Cref_max
    
    return Cref_min, Cref_max, Dref_min, Dref_max


def parse_input(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    idx = 0
    num_test_cases = int(lines[idx])
    idx += 1

    testcases = []

    for _ in range(num_test_cases):
        assert lines[idx].startswith("TestCase")
        testcase_name = lines[idx]
        idx += 1
        num_dags = int(lines[idx])
        idx += 1
        dag_list = []
        for _ in range(num_dags):
            dag_header = lines[idx].split()
            assert dag_header[0] == "DAG"
            dag_id = int(dag_header[1])
            num_subtasks = int(dag_header[2])
            idx += 1
            subtasks = []
            for _ in range(num_subtasks):
                parts = lines[idx].split()
                operation = parts[0]
                t, n, x, b = map(int, parts[1:5])
                resources = parts[5].split(',') if len(parts) > 5 else []
                subtasks.append({
                    'id': len(subtasks)+1,
                    'operation': operation,
                    't': t,
                    'n': n,
                    'x': x,
                    'b': b,
                    'resources': set(resources)
                })
                idx += 1
            num_edges = int(lines[idx])
            idx += 1
            edges = []
            for _ in range(num_edges):
                u, v = map(int, lines[idx].split())
                edges.append((u, v))
                idx += 1
            dag_list.append({
                'dag_id': dag_id,
                'subtasks': subtasks,
                'edges': edges
            })
        num_ecus = int(lines[idx])
        idx += 1
        ecus = []
        for _ in range(num_ecus):
            parts = lines[idx].split()
            ecu_id = parts[0]
            sensor_count = int(parts[1])
            cpu_count = int(parts[2])
            actuator_count = int(parts[3])
            resources = set(parts[4].split(','))
            cores = cpu_count
            ecus.append({
                'id': ecu_id,
                'cores': cores,
                'sensor_count': sensor_count,
                'cpu_count': cpu_count,
                'actuator_count': actuator_count,
                'resources': resources
            })
            idx += 1
        testcases.append({
            'name': testcase_name,
            'dags': dag_list,
            'ecus': ecus
        })
    return testcases

class Chromosome:
    def __init__(self, subtask_count, ecu_count, ecu_cores):
        self.assignment=[[] for _ in range(subtask_count)]  
        self.fitness=None
        self.makespan=None
        self.loads=[0] * ecu_count
        self.schedule=None
        self.ecu_cores=ecu_cores
        self.Lmax=None
        self.Lmin=None
        self.load_imbalance=None

    def copy(self):
        c=Chromosome(len(self.assignment), len(self.loads), self.ecu_cores)
        c.assignment=[a[:] for a in self.assignment]
        c.fitness=self.fitness
        c.makespan=self.makespan
        c.loads=self.loads[:]
        c.schedule=None if self.schedule is None else self.schedule.copy()
        c.Lmax=self.Lmax
        c.Lmin=self.Lmin
        c.load_imbalance=self.load_imbalance
        return c


def find_critical_path(schedule, dag):
    n=len(schedule)
    edges=dag['edges']
    adj=[[] for _ in range(n)]
    rev=[[] for _ in range(n)]
    for u, v in edges:
        adj[u-1].append(v-1)
        rev[v-1].append(u-1)
    sinks=[i for i in range(n) if not adj[i]]
    def dfs(u):
        if not rev[u]:
            return [u]
        max_path=[]
        for pred in rev[u]:
            path=dfs(pred)
            if sum(schedule[x][1]-schedule[x][0] for x in path) > sum(schedule[x][1]-schedule[x][0] for x in max_path):
                max_path=path
        return max_path+[u]
    max_finish=-1
    max_sink=-1
    for s in sinks:
        if schedule[s][1] > max_finish:
            max_finish=schedule[s][1]
            max_sink=s
    return dfs(max_sink)
    
def local_search(chrom, dag, ecus, inter_ecu_delay=0):
    improved=True
    while improved:
        improved=False
        result=schedule_chromosome(chrom, dag, ecus, inter_ecu_delay)
        if result is None:
            break
        schedule, makespan, loads, load_stddev=result
        path=find_critical_path(schedule, dag)
        for idx in path:
            best_assignment=chrom.assignment[idx]
            best_makespan=makespan
            valid=get_valid_ecu_core(dag['subtasks'][idx], ecus)
            for ecu_idx, core_idx in valid:
                if (ecu_idx, core_idx)==chrom.assignment[idx]:
                    continue
                orig_assignment=chrom.assignment[idx]
                chrom.assignment[idx]=(ecu_idx, core_idx)
                result2=schedule_chromosome(chrom, dag, ecus, inter_ecu_delay)
                if result2 is not None:
                    _, new_makespan, _, _=result2
                    if new_makespan<best_makespan:
                        best_assignment=(ecu_idx, core_idx)
                        best_makespan=new_makespan
                        improved=True
                chrom.assignment[idx]=orig_assignment
            chrom.assignment[idx]=best_assignment
    return chrom

def get_valid_ecu_core(subtask, ecus):
    valid=[]
    for ecu_idx, ecu in enumerate(ecus):
        if subtask['resources'].issubset(ecu['resources']):
            for core in range(ecu['cores']):
                valid.append((ecu_idx, core))
    return valid


def topological_order(subtask_count, edges):
    adj=[[] for _ in range(subtask_count)]
    indeg=[0] * subtask_count
    for u, v in edges:
        adj[u-1].append(v-1)
        indeg[v-1]+=1
    q=deque([i for i in range(subtask_count) if indeg[i]==0])
    order=[]
    while q:
        u=q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v]-=1
            if indeg[v]==0:
                q.append(v)
    return order if len(order)==subtask_count else None


def schedule_chromosome(chrom, dag, ecus, delay_matrix=None):
    subtasks=dag['subtasks']
    edges=dag['edges']
    subtask_count=len(subtasks)
    ecu_count=len(ecus)
    order=topological_order(subtask_count, edges)
    if order is None:
        return None

    core_timeline=[[0 for _ in range(ecu['cores'])] for ecu in ecus]
    schedule=[(-1,-1) for _ in range(subtask_count)]
    assignment=chrom.assignment
    finish_times=[0] * subtask_count

    for idx in order:
        subtask=subtasks[idx]
        assigned=assignment[idx]
        # Defensive: always treat as list of (ecu_idx, core_idx)
        if isinstance(assigned, tuple):
            assigned=[assigned]
        elif isinstance(assigned, int):
            assigned=[(assigned, 0)]
        # Earliest start: all predecessors finish+inter-ECU delay if needed
        earliest=0
        for u, v in edges:
            if v-1==idx:
                pred_idx=u-1
                pred_assigned=assignment[pred_idx]
                if isinstance(pred_assigned, tuple):
                    pred_assigned=[pred_assigned]
                elif isinstance(pred_assigned, int):
                    pred_assigned=[(pred_assigned, 0)]
                pred_finish=finish_times[pred_idx]
                max_delay=0
                for (pred_ecu, _) in pred_assigned:
                    for (this_ecu, _) in assigned:
                        if delay_matrix:
                            d=delay_matrix[pred_ecu][this_ecu] if pred_ecu !=this_ecu else 0
                        else:
                            d=0
                        if d > max_delay:
                            max_delay=d
                earliest=max(earliest, pred_finish+max_delay)
        # All assigned ECUs/cores must be available at the same time
        assigned=to_assignment_list(assignment[idx])
        for (ecu_idx, core_idx) in assigned:
            earliest=max(earliest, core_timeline[ecu_idx][core_idx])
        x, n, t=subtask['x'], subtask['n'], subtask['t']
        exec_time=math.ceil(x/n) * t
        start=earliest
        finish=start+exec_time
        schedule[idx]=(start, finish)
        finish_times[idx]=finish
        assigned=to_assignment_list(assignment[idx])
        for (ecu_idx, core_idx) in assigned:
            core_timeline[ecu_idx][core_idx]=finish

    makespan=max(finish for start, finish in schedule)
    loads=[0] * ecu_count
    for idx, assigned in enumerate(assignment):
        exec_time=schedule[idx][1]-schedule[idx][0]
        # Defensive: always treat as list
        if isinstance(assigned, tuple):
            assigned=[assigned]
        elif isinstance(assigned, int):
            assigned=[(assigned, 0)]
        for (ecu_idx, _) in assigned:
            loads[ecu_idx]+=exec_time
    return schedule, makespan, loads, None

def check_assignments(population):
    for chrom in population:
        for a in chrom.assignment:
            assert isinstance(a, list)
            assert all(isinstance(pair, tuple) and len(pair)==2 for pair in a)


def is_valid_assignment(chrom, dag, ecus):
    for idx, subtask in enumerate(dag['subtasks']):
        for ecu_idx, _ in to_assignment_list(chrom.assignment[idx]):
            if ecu_idx < 0 or not subtask['resources'].issubset(ecus[ecu_idx]['resources']):
                return False
    return True

def compute_covers_cache(subtasks, ecus):
    covers_cache = {}
    for idx, subtask in enumerate(subtasks):
        covers_cache[idx] = list(find_minimal_ecu_covers(subtask, ecus))
    return covers_cache


def initialize_population(pop_size, dag, ecus, covers_cache):
    subtask_count = len(dag['subtasks'])
    ecu_count = len(ecus)
    ecu_cores = [ecu['cores'] for ecu in ecus]
    population = []
    for _ in range(pop_size):
        chrom = Chromosome(subtask_count, ecu_count, ecu_cores)
        for idx, subtask in enumerate(dag['subtasks']):
            covers = covers_cache[idx]
            if not covers:
                raise ValueError(f"Subtask {idx+1} cannot be scheduled on any ECU combination.")
            cover = random.choice(covers)
            assignment = []
            for ecu_idx in cover:
                core = random.choice(range(ecus[ecu_idx]['cores']))
                assignment.append((ecu_idx, core))
            chrom.assignment[idx] = assignment
        population.append(chrom)
    return population


def tournament_selection(population, k=3):
    selected=random.sample(population, k)
    return min(selected, key=lambda c: c.fitness)

def evaluate_population(population, dag, ecus, alpha=0.5, beta=0.5, 
                        Cref_min=0, Cref_max=1, Dref_min=0, Dref_max=1, eps=1e-5):
    for chrom in population:
        result=schedule_chromosome(chrom, dag, ecus)
        if result is None:
            chrom.fitness=float('inf')
            continue
        schedule, makespan, loads, _=result
        chrom.schedule=schedule
        chrom.makespan=makespan
        chrom.loads=loads
        chrom.Lmax=max(loads)
        chrom.Lmin=min(loads)
        chrom.load_imbalance=chrom.Lmax-chrom.Lmin
    for chrom in population:
        if chrom.fitness==float('inf'):
            continue
        norm_makespan=(chrom.makespan-Cref_min)/(Cref_max-Cref_min+eps)
        norm_load_imb=(chrom.load_imbalance-Dref_min)/(Dref_max-Dref_min+eps)
        chrom.fitness=alpha * norm_makespan+beta * norm_load_imb
    return [chrom.fitness for chrom in population]


def assignment_crossover(parent1, parent2, dag, ecus, covers_cache):
    child=parent1.copy()
    for idx in range(len(parent1.assignment)):
        p1=to_assignment_list(parent1.assignment[idx])
        p2=to_assignment_list(parent2.assignment[idx])
        child.assignment[idx]=[pair for pair in random.choice([p1, p2])]
        # covers=list(find_minimal_ecu_covers(dag['subtasks'][idx], ecus))
        covers = covers_cache[idx]

        assigned_ecus=set(ecu_idx for ecu_idx, _ in child.assignment[idx])
        valid=False
        for cover in covers:
            if set(cover)==assigned_ecus:
                valid=True
                break
        if not valid:
            cover=random.choice(covers)
            assignment=[]
            for ecu_idx in cover:
                core=random.choice(range(ecus[ecu_idx]['cores']))
                assignment.append((ecu_idx, core))
            child.assignment[idx]=assignment
    return child

def mutate(chrom, dag, ecus, covers_cache, mutation_rate=0.3):
    for idx, subtask in enumerate(dag['subtasks']):
        if random.random()<mutation_rate:
            covers=list(find_minimal_ecu_covers(subtask, ecus))
            current_cover=set(ecu_idx for ecu_idx, _ in to_assignment_list(chrom.assignment[idx]))
            options=[cover for cover in covers if set(cover) !=current_cover]
            if options:
                cover=random.choice(options)
                assignment=[]
                for ecu_idx in cover:
                    core=random.choice(range(ecus[ecu_idx]['cores']))
                    assignment.append((ecu_idx, core))
                chrom.assignment[idx]=assignment


def genetic_algorithm(dag, ecus,  covers_cache,  delay_matrix=DELAY_MATRIX, pop_size=100, generations=200, alpha=0.5, beta=0.5, elitism=True, Cref_min=0, Cref_max=1, Dref_min=0, Dref_max=1):
    population = initialize_population(pop_size, dag, ecus, covers_cache)
    evaluate_population(population, dag, ecus, alpha, beta, Cref_min, Cref_max, Dref_min, Dref_max)
    best = min(population, key=lambda c: c.fitness)
    no_improvement_count = 0
    last_best_fitness = None

    for gen in range(generations):
        new_population = []
        if elitism:
            new_population.append(best.copy())
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = assignment_crossover(parent1, parent2, dag, ecus, covers_cache)
            mutate(child, dag, ecus, covers_cache, 0.2)
            # REMOVE or comment out local search here for speed
            # child = local_search(child, dag, ecus)
            new_population.append(child)
        fitnesses = evaluate_population(new_population, dag, ecus, alpha, beta, Cref_min, Cref_max, Dref_min, Dref_max)
        population = new_population
        current_best = min(population, key=lambda c: c.fitness)
        if current_best.fitness < best.fitness:
            best = current_best.copy()
        if current_best.fitness == last_best_fitness:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        last_best_fitness = current_best.fitness
        if no_improvement_count >= 10:
            print("No improvement for 10 generations, stopping early.", flush=True)
            break

    # Run local search only on the best solution at the end (optional)
    best = local_search(best, dag, ecus)
    return best

def run_case(case):
    import time, tracemalloc  # Needed for each process
    start_time = time.time()
    tracemalloc.start()
    dag_list = case['dags']
    ecus = case['ecus']
    merged_subtasks, merged_edges = merge_dags(dag_list)
    combined_dag = {
        'dag_id': 0,
        'subtasks': merged_subtasks,
        'edges': merged_edges
    }
    Cref_min, Cref_max, Dref_min, Dref_max = compute_theoretical_bounds(dag_list)
    covers_cache = compute_covers_cache(combined_dag['subtasks'], ecus)
    best = genetic_algorithm(
        combined_dag, ecus, covers_cache=covers_cache, delay_matrix=DELAY_MATRIX,
        pop_size=20, generations=50, alpha=0.5, beta=0.5, elitism=True,
        Cref_min=Cref_min, Cref_max=Cref_max, Dref_min=Dref_min, Dref_max=Dref_max
    )
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    memory_used = peak / 1024 / 1024  # MB

    # Prepare summary result for this case
    result = {
        'name': case['name'],
        'makespan': best.makespan,
        'Lmax': max(best.loads),
        'Lmin': min(best.loads),
        'Cref_min': Cref_min,
        'Cref_max': Cref_max,
        'Dref_min': Dref_min,
        'Dref_max': Dref_max,
        'load_imbalance': max(best.loads) - min(best.loads),
        'objective': alpha * ((best.makespan-Cref_min)/(Cref_max-Cref_min+1e-5)) +
                     beta * ((max(best.loads)-min(best.loads)-Dref_min)/(Dref_max-Dref_min+1e-5)),
        'elapsed_time': elapsed_time,
        'memory_used': memory_used,
        'schedule': best.schedule,
        'assignments': best.assignment,
        'subtasks': combined_dag['subtasks'],
        'ecus': ecus
    }
    return result

if __name__ == "__main__":
    print("Started parsing...", flush=True)
    testcases = parse_input("input_30.txt")
    print("Finished parsing.", flush=True)
    num_cases = len(testcases)

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(run_case, testcases):
            # Print or save per-case results as they complete
            print(f"=== {res['name']} ===", flush=True)
            print(f"Makespan: {res['makespan']}", flush=True)
            # print(f"Lmax: {res['Lmax']}", flush=True)
            # print(f"Lmin: {res['Lmin']}", flush=True)
            # print(f"Cref_min: {res['Cref_min']}", flush=True)
            # print(f"Cref_max: {res['Cref_max']}", flush=True)
            # print(f"Dref_min: {res['Dref_min']}", flush=True)
            # print(f"Dref_max: {res['Dref_max']}", flush=True)
            print("\nSchedule:", flush=True)
            for idx, subtask in enumerate(res['subtasks']):
                start, finish = res['schedule'][idx]
                for ecu_idx, core_idx in to_assignment_list(res['assignments'][idx]):
                    print(
                        f"Subtask {idx+1} ({subtask['operation']}): "
                        f"ECU {res['ecus'][ecu_idx]['id']} Core {core_idx+1}, "
                        f"Start {start}, Finish {finish}",
                        flush=True
                    )

            print(f"Load Imbalance: {res['load_imbalance']}", flush=True)
            print(f"Objective Function Value: {res['objective']:.4f}", flush=True)
            print(f"Elapsed Time (this testcase): {res['elapsed_time']:.2f} seconds", flush=True)
            print(f"Peak Memory Usage (this testcase): {res['memory_used']:.3f} MB", flush=True)
            results.append(res)

    # Compute averages
    avg_time = sum(r['elapsed_time'] for r in results) / num_cases
    avg_mem = sum(r['memory_used'] for r in results) / num_cases
    avg_objective = sum(r['objective'] for r in results) / num_cases
    avg_makespan = sum(r['makespan'] for r in results) / num_cases
    avg_load_imbalance = sum(r['load_imbalance'] for r in results) / num_cases
    print(f"Average Load Imbalance: {avg_load_imbalance:.2f}", flush=True)

    print(f"\nAverage Time per Testcase: {avg_time:.2f} seconds", flush=True)
    print(f"Average Peak Memory Usage per Testcase: {avg_mem:.3f} MB", flush=True)
    print(f"Average Objective Function Value: {avg_objective:.4f}", flush=True)
    print(f"Average Makespan: {avg_makespan:.2f}", flush=True)
