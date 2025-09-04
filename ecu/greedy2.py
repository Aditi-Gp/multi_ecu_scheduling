import math
from collections import deque
import itertools
import copy

alpha = 0.7
beta = 0.3




def read_input(filename):
    with open(filename,'r') as f:
        lines=[line.strip() for line in f if line.strip()]
    idx=0
    T=int(lines[idx]); idx+=1
    testcases=[]
    for _ in range(T):
        testcase_name=lines[idx]; idx+=1
        dag_count=int(lines[idx]); idx+=1
        dag_list=[]
        for _ in range(dag_count):
            dag_header=lines[idx].split(); idx+=1
            dag_id=int(dag_header[1])
            subtask_count=int(dag_header[2])
            subtasks=[]
            for _ in range(subtask_count):
                parts=lines[idx].split(); idx+=1
                op_id=parts[0]
                t,n,x,b=map(int,parts[1:5])
                resources=set(parts[5].split(','))
                subtasks.append({'id': op_id,'t': t,'n': n,'x': x,'b': b,'resources': resources})
            edge_count=int(lines[idx]); idx+=1
            edges=[]
            for _ in range(edge_count):
                u,v=map(int,lines[idx].split()); idx+=1
                edges.append((u,v))
            dag_list.append({'dag_id': dag_id,'subtasks': subtasks,'edges': edges})
        ecu_count=int(lines[idx]); idx+=1
        ecus=[]
        for _ in range(ecu_count):
            parts=lines[idx].split(); idx+=1
            ecu_id=parts[0]
            cores=int(parts[1])
            resources=set(parts[4].split(','))
            ecus.append({'id': ecu_id,'cores': cores,'resources': resources})
        testcases.append({'name': testcase_name,'dags': dag_list,'ecus': ecus})
    return testcases



def merge_dags(dag_list):
    merged_subtasks=[]
    merged_edges=[]
    offset=0
    for dag in dag_list:
        n_subtasks=len(dag['subtasks'])
        merged_subtasks.extend(dag['subtasks'])
        for u,v in dag['edges']:
            merged_edges.append((u+offset,v+offset))
        offset+=n_subtasks
    return {'subtasks': merged_subtasks,'edges': merged_edges}

def topological_order(subtask_count, edges):
    adj = [[] for _ in range(subtask_count)]
    indeg = [0] * subtask_count
    for u, v in edges:
        adj[u-1].append(v-1)
        indeg[v-1] += 1
    q = deque([i for i in range(subtask_count) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == subtask_count else None

def find_minimal_ecu_covers(subtask, ecus):
    required = subtask['resources']
    ecu_indices = list(range(len(ecus)))
    for r in range(1, len(ecus)+1):
        for combo in itertools.combinations(ecu_indices, r):
            combined = set()
            for idx in combo:
                combined |= ecus[idx]['resources']
            if required.issubset(combined):
                # Minimality check
                minimal = True
                for i in range(r):
                    test_combo = combo[:i] + combo[i+1:]
                    test_combined = set()
                    for idx in test_combo:
                        test_combined |= ecus[idx]['resources']
                    if required.issubset(test_combined):
                        minimal = False
                        break
                if minimal:
                    yield combo

def subtask_exec_time(subtask):
    return math.ceil(subtask['x']/subtask['n']) * subtask['t']

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
        exec_time = subtask_exec_time(subtasks[u])
        for v in adj[u]:
            dp[v] = max(dp[v], dp[u] + exec_time)
    for u in range(subtask_count):
        exec_time = subtask_exec_time(subtasks[u])
        dp[u] += exec_time
    return max(dp)

def cal_workload(dag):
    return sum(subtask_exec_time(st) for st in dag['subtasks'])

def get_critical_path_indices(dag):
    # Returns the indices of subtasks on the critical path
    subtasks = dag['subtasks']
    edges = dag['edges']
    subtask_count = len(subtasks)
    adj = [[] for _ in range(subtask_count)]
    rev = [[] for _ in range(subtask_count)]
    for u, v in edges:
        adj[u-1].append(v-1)
        rev[v-1].append(u-1)
    order = topological_order(subtask_count, edges)
    dp = [0] * subtask_count
    pred = [-1] * subtask_count
    for u in order:
        exec_time = subtask_exec_time(subtasks[u])
        for v in adj[u]:
            if dp[v] < dp[u] + exec_time:
                dp[v] = dp[u] + exec_time
                pred[v] = u
    # Find the node with the max dp
    end = max(range(subtask_count), key=lambda i: dp[i])
    # Backtrack to get the path
    path = []
    while end != -1:
        path.append(end)
        end = pred[end]
    return list(reversed(path))

def greedy_schedule_critical_path_first(dag, ecus, delay_matrix=None):
    subtasks = dag['subtasks']
    edges = dag['edges']
    subtask_count = len(subtasks)
    ecu_count = len(ecus)
    order = topological_order(subtask_count, edges)
    # Compute critical path indices
    cp_indices = set(get_critical_path_indices(dag))
    # Sort ready tasks: critical-path subtasks first
    order = sorted(order, key=lambda idx: (0 if idx in cp_indices else 1, idx))
    core_timeline = [[0 for _ in range(ecu['cores'])] for ecu in ecus]
    assignment = [None for _ in range(subtask_count)]
    schedule = [(-1, -1) for _ in range(subtask_count)]
    finish_times = [0] * subtask_count

    for idx in order:
        subtask = subtasks[idx]
        covers = list(find_minimal_ecu_covers(subtask, ecus))
        best_start, best_finish, best_assignment = None, None, None
        for cover in covers:
            core_choices = [list(range(ecus[ecu_idx]['cores'])) for ecu_idx in cover]
            for core_combo in itertools.product(*core_choices):
                assigned = list(zip(cover, core_combo))
                earliest = 0
                for u, v in edges:
                    if v-1 == idx:
                        pred_idx = u-1
                        # pred_assigned = assignment[pred_idx]
                        pred_assigned = assignment[pred_idx]
                        if pred_assigned is None:
                            pred_assigned = []
                        pred_finish = finish_times[pred_idx]
                        max_delay = 0
                        for (pred_ecu, _) in pred_assigned:
                            for (this_ecu, _) in assigned:
                                if delay_matrix:
                                    d = delay_matrix[pred_ecu][this_ecu] if pred_ecu != this_ecu else 0
                                else:
                                    d = 0
                                if d > max_delay:
                                    max_delay = d
                        earliest = max(earliest, pred_finish + max_delay)
                for (ecu_idx, core_idx) in assigned:
                    earliest = max(earliest, core_timeline[ecu_idx][core_idx])
                exec_time = subtask_exec_time(subtask)
                start = earliest
                finish = start + exec_time
                # Greedy: earliest finish, then best load
                if (best_finish is None) or (finish < best_finish) or (finish == best_finish and sum(core_timeline[ecu_idx][core_idx] for (ecu_idx, core_idx) in assigned) < sum(core_timeline[ecu_idx][core_idx] for (ecu_idx, core_idx) in best_assignment or [])):
                    best_start, best_finish, best_assignment = start, finish, assigned

        assignment[idx] = best_assignment
        schedule[idx] = (best_start, best_finish)
        finish_times[idx] = best_finish
        for (ecu_idx, core_idx) in best_assignment:
            core_timeline[ecu_idx][core_idx] = best_finish

    makespan = max(finish for start, finish in schedule)
    loads = [0] * ecu_count
    for idx, assigned in enumerate(assignment):
        exec_time = schedule[idx][1] - schedule[idx][0]
        for (ecu_idx, _) in assigned:
            loads[ecu_idx] += exec_time
    return assignment, schedule, makespan, loads

def local_search_swap(assignment, schedule, dag, ecus, delay_matrix=None, max_iter=100):
    # Try swapping assignments of pairs of subtasks to improve makespan
    best_assignment = copy.deepcopy(assignment)
    best_schedule = copy.deepcopy(schedule)
    best_makespan = max(finish for start, finish in best_schedule)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        for i in range(len(assignment)):
            for j in range(i+1, len(assignment)):
                # Swap assignments
                new_assignment = copy.deepcopy(best_assignment)
                new_assignment[i], new_assignment[j] = new_assignment[j], new_assignment[i]
                # Recompute schedule
                sched, makespan, loads, _ = greedy_schedule_from_assignment(new_assignment, dag, ecus, delay_matrix)

                if makespan < best_makespan:
                    best_assignment = new_assignment
                    best_schedule = sched
                    best_makespan = makespan
                    improved = True
        iter_count += 1
    return best_assignment, best_schedule

def greedy_schedule_from_assignment(assignment, dag, ecus, delay_matrix=None):
    # Given an assignment, compute the schedule, makespan, loads
    subtasks = dag['subtasks']
    edges = dag['edges']
    subtask_count = len(subtasks)
    ecu_count = len(ecus)
    core_timeline = [[0 for _ in range(ecu['cores'])] for ecu in ecus]
    schedule = [(-1, -1) for _ in range(subtask_count)]
    finish_times = [0] * subtask_count
    order = topological_order(subtask_count, edges)
    for idx in order:
        subtask = subtasks[idx]
        assigned = assignment[idx]
        earliest = 0
        for u, v in edges:
            if v-1 == idx:
                pred_idx = u-1
                pred_assigned = assignment[pred_idx]
                pred_finish = finish_times[pred_idx]
                max_delay = 0
                for (pred_ecu, _) in pred_assigned:
                    for (this_ecu, _) in assigned:
                        if delay_matrix:
                            d = delay_matrix[pred_ecu][this_ecu] if pred_ecu != this_ecu else 0
                        else:
                            d = 0
                        if d > max_delay:
                            max_delay = d
                earliest = max(earliest, pred_finish + max_delay)
        for (ecu_idx, core_idx) in assigned:
            earliest = max(earliest, core_timeline[ecu_idx][core_idx])
        exec_time = subtask_exec_time(subtask)
        start = earliest
        finish = start + exec_time
        schedule[idx] = (start, finish)
        finish_times[idx] = finish
        for (ecu_idx, core_idx) in assigned:
            core_timeline[ecu_idx][core_idx] = finish
    makespan = max(finish for start, finish in schedule)
    loads = [0] * ecu_count
    for idx, assigned in enumerate(assignment):
        exec_time = schedule[idx][1] - schedule[idx][0]
        for (ecu_idx, _) in assigned:
            loads[ecu_idx] += exec_time
    return schedule, makespan, loads, None

# def main():
#     # Read your input as before
#     testcases = read_input('input.txt')
#     for case in testcases:
#         print(f"{case['name']}")
#         dag = merge_dags(case['dags'])
#         ecus = case['ecus']
#         assignment, schedule, makespan, loads = greedy_schedule_critical_path_first(dag, ecus)
#         # Local search improvement
#         assignment, schedule = local_search_swap(assignment, schedule, dag, ecus)
#         for idx, assigned in enumerate(assignment):
#             start, finish = schedule[idx]
#             for ecu_idx, core_idx in assigned:
#                 print(f"Subtask {idx+1}: ECU {ecus[ecu_idx]['id']} Core {core_idx+1}, Start {start}, Finish {finish}")
#         print("Makespan:", makespan)
#         print("ECU Loads:", loads)
#         print("Lmax:", max(loads), "Lmin:", min(loads))
#         Cref_min = compute_critical_path(dag)
#         Cref_max = cal_workload(dag)
#         Dref_min = 0
#         Dref_max = Cref_max
#         print(f"Cref_min: {Cref_min}")
#         print(f"Cref_max : {Cref_max}")
#         print(f"Dref_min: {Dref_min}")
#         print(f"Dref_max : {Dref_max}")
#         norm_makespan = (makespan - Cref_min) / (Cref_max - Cref_min + 1e-5)
#         norm_load_imb = (max(loads) - min(loads) - Dref_min) / (Dref_max - Dref_min + 1e-5)
#         objective = alpha * norm_makespan + beta * norm_load_imb
#         print(f"Normalized Makespan: {norm_makespan:.4f}")
#         print(f"Normalized Load Imbalance: {norm_load_imb:.4f}")
#         print(f"Objective Function Value: {objective:.4f}")

# if __name__ == '__main__':
#     main()


import concurrent.futures

def run_case_greedy(case):
    import time, tracemalloc
    start_time = time.time()
    tracemalloc.start()
    dag = merge_dags(case['dags'])
    ecus = case['ecus']
    assignment, schedule, makespan, loads = greedy_schedule_critical_path_first(dag, ecus)
    # Optionally, add local search:
    # assignment, schedule = local_search_swap(assignment, schedule, dag, ecus)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    memory_used = peak / 1024 / 1024  # MB

    Lmax = max(loads)
    Lmin = min(loads)
    load_imbalance = Lmax - Lmin
    Cref_min = compute_critical_path(dag)
    Cref_max = cal_workload(dag)
    Dref_min = 0
    Dref_max = Cref_max

    norm_makespan = (makespan - Cref_min) / (Cref_max - Cref_min + 1e-5)
    norm_load_imb = (Lmax - Lmin - Dref_min) / (Dref_max - Dref_min + 1e-5)
    objective = alpha * norm_makespan + beta * norm_load_imb

    # Prepare result dict
    result = {
        'name': case['name'],
        'makespan': makespan,
        'Lmax': Lmax,
        'Lmin': Lmin,
        'Cref_min': Cref_min,
        'Cref_max': Cref_max,
        'Dref_min': Dref_min,
        'Dref_max': Dref_max,
        'load_imbalance': load_imbalance,
        'objective': objective,
        'elapsed_time': elapsed_time,
        'memory_used': memory_used,
        'schedule': schedule,
        'assignments': assignment,
        'subtasks': dag['subtasks'],
        'ecus': ecus,
        'loads': loads,
    }
    return result

def to_assignment_list(assignment):
    # Defensive: always treat as list of (ecu_idx, core_idx)
    if isinstance(assignment, list):
        return assignment
    elif isinstance(assignment, tuple):
        return [assignment]
    elif isinstance(assignment, int):
        return [(assignment, 0)]
    else:
        raise ValueError("Invalid assignment type")

if __name__ == "__main__":
    print("Started parsing...", flush=True)
    testcases = read_input("input_25.txt")
    print("Finished parsing.", flush=True)
    num_cases = len(testcases)

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(run_case_greedy, testcases):
            print(f"=== {res['name']} ===", flush=True)
            print(f"Makespan: {res['makespan']}", flush=True)
            print("\nSchedule:", flush=True)
            for idx, subtask in enumerate(res['subtasks']):
                start, finish = res['schedule'][idx]
                # If you want operation name, use subtask['id'] or subtask['operation'] if available
                for ecu_idx, core_idx in to_assignment_list(res['assignments'][idx]):
                    print(
                        f"Subtask {idx+1} ({subtask['id']}): "
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
