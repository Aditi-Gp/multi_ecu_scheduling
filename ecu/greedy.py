import math
from collections import deque
import itertools

alpha=0.5
beta=0.5



"""
Input->setof merged dags, ecu, inter
output->assignment of each subtask to Ecu+cores, start&finish time, makespan, ecu loads, objective function value

Greedy RUle: At each scheduling step, for each ready subtask (i.e., all dependencies are satisfied), assign it to the available ECU/core(s) (or minimal ECU/core set) that allows the subtask to finish at the earliest possible time.
If there are ties, select the assignment that best balances the current ECU loads.
Primary greedy criterion:
Earliest finish time for the subtask, considering all valid ECU/core assignments, resource constraints, and inter-ECU delays.
Secondary (tie-breaker) criterion:
Among assignments with the same earliest finish, choose the one that minimizes the maximum ECU load (or best balances the load).

Algorithm:
    1. Topological sort: for all subtasks across all dags
    2. maintain the earliest time at which ewch ecu and core is available
    3. for each subtask in topological order:
            find minimal covers for all ecus.
            for each possible combination of cores for the chosen ECUs (one core per ECU in the cover):
                Compute the earliest time all required ECUs/cores are simultaneously available.
                for each predecessor subtask, if it is assigned to a different ECU, add the inter-ECU delay to the finish time of the predecessor.
                The subtask cannot start before all its predecessors have finished (plus any required delays).
            apply greedy rule
            update schedules, load
    4. after all subtasks scheduled-> calculate makespan, ecu ;oads, lmax, lmin
    5. normalise and calculate objective function
"""


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

def topological_order(subtask_count,edges):
    adj=[[] for _ in range(subtask_count)]
    indeg=[0]*subtask_count
    for u,v in edges:
        adj[u-1].append(v-1)
        indeg[v-1]+=1
    q=deque([i for i in range(subtask_count) if indeg[i]==0])
    order=[]
    while q:
        u=q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v]==0:
                q.append(v)
    return order if len(order)==subtask_count else None

def find_minimal_ecu_covers(subtask,ecus):
    required=subtask['resources']
    ecu_indices=list(range(len(ecus)))
    for r in range(1,len(ecus)+1):
        for combo in itertools.combinations(ecu_indices,r):
            combined=set()
            for idx in combo:
                combined|=ecus[idx]['resources']
            if required.issubset(combined):
                # Minimality check
                minimal=True
                for i in range(r):
                    test_combo=combo[:i]+combo[i+1:]
                    test_combined=set()
                    for idx in test_combo:
                        test_combined|=ecus[idx]['resources']
                    if required.issubset(test_combined):
                        minimal=False
                        break
                if minimal:
                    yield combo

def fibd_critical_path(dag):
    subtasks=dag['subtasks']
    edges=dag['edges']
    subtask_count=len(subtasks)
    adj=[[] for _ in range(subtask_count)]
    for u,v in edges:
        adj[u-1].append(v-1)
    # DP for longest path
    dp=[0]*subtask_count
    order=topological_order(subtask_count,edges)
    for u in order:
        x,n,t=subtasks[u]['x'],subtasks[u]['n'],subtasks[u]['t']
        exec_time=math.ceil(x/n)*t
        for v in adj[u]:
            dp[v]=max(dp[v],dp[u]+exec_time)
    for u in range(subtask_count):
        x,n,t=subtasks[u]['x'],subtasks[u]['n'],subtasks[u]['t']
        exec_time=math.ceil(x/n)*t
        dp[u]+=exec_time
    return max(dp)

def cal_workload(dag):
    subtasks=dag['subtasks']
    total=0
    for subtask in subtasks:
        x,n,t=subtask['x'],subtask['n'],subtask['t']
        total+=math.ceil(x/n)*t
    return total


def greedy_schedule(dag,ecus,delay_matrix=None):
    subtasks=dag['subtasks']
    edges=dag['edges']
    subtask_count=len(subtasks)
    ecu_count=len(ecus)
    order=topological_order(subtask_count,edges)
    core_timeline=[[0 for _ in range(ecu['cores'])] for ecu in ecus]
    assignment=[[] for _ in range(subtask_count)]
    schedule=[(-1,-1) for _ in range(subtask_count)]
    finish_times=[0]*subtask_count

    for idx in order:
        subtask=subtasks[idx]
        covers=list(find_minimal_ecu_covers(subtask,ecus))
        best_start,best_finish,best_assignment=None,None,None

        for cover in covers:
            import itertools
            core_choices=[list(range(ecus[ecu_idx]['cores'])) for ecu_idx in cover]
            for core_combo in itertools.product(*core_choices):
                assigned=list(zip(cover,core_combo))
                earliest=0
                for u,v in edges:
                    if v-1==idx:
                        pred_idx=u-1
                        pred_assigned=assignment[pred_idx]
                        pred_finish=finish_times[pred_idx]
                        max_delay=0
                        for (pred_ecu,_) in pred_assigned:
                            for (this_ecu,_) in assigned:
                                if delay_matrix:
                                    d=delay_matrix[pred_ecu][this_ecu] if pred_ecu != this_ecu else 0
                                else:
                                    d=0
                                if d>max_delay:
                                    max_delay=d
                        earliest=max(earliest,pred_finish+max_delay)
                for (ecu_idx,core_idx) in assigned:
                    earliest=max(earliest,core_timeline[ecu_idx][core_idx])
                x,n,t=subtask['x'],subtask['n'],subtask['t']
                exec_time=math.ceil(x/n)*t
                start=earliest
                finish=start+exec_time
                # Greedy: earliest finish,then best load
                if (best_finish is None) or (finish<best_finish) or (finish==best_finish and sum(core_timeline[ecu_idx][core_idx] for (ecu_idx,core_idx) in assigned)<sum(core_timeline[ecu_idx][core_idx] for (ecu_idx,core_idx) in best_assignment)):
                    best_start,best_finish,best_assignment=start,finish,assigned

        assignment[idx]=best_assignment
        schedule[idx]=(best_start,best_finish)
        finish_times[idx]=best_finish
        for (ecu_idx,core_idx) in best_assignment:
            core_timeline[ecu_idx][core_idx]=best_finish

    makespan=max(finish for start,finish in schedule)
    loads=[0]*ecu_count
    for idx,assigned in enumerate(assignment):
        exec_time=schedule[idx][1]-schedule[idx][0]
        for (ecu_idx,_) in assigned:
            loads[ecu_idx]+=exec_time
    return assignment,schedule,makespan,loads

def main():
    testcases=read_input('input.txt')
    for case in testcases:
        print(f"{case['name']}")
        dag=merge_dags(case['dags'])
        ecus=case['ecus']
        assignment,schedule,makespan,loads=greedy_schedule(dag,ecus)
        for idx,assigned in enumerate(assignment):
            start,finish=schedule[idx]
            for ecu_idx,core_idx in assigned:
                print(f"Subtask {idx+1}: ECU {ecus[ecu_idx]['id']} Core {core_idx+1},Start {start},Finish {finish}")
        print("Makespan:",makespan)
        print("ECU Loads:",loads)
        print("Lmax:",max(loads),"Lmin:",min(loads))
        Cref_min=fibd_critical_path(dag)
        Cref_max=cal_workload(dag)
        Dref_min=0
        Dref_max=Cref_max
        print(f"Cref_min: {Cref_min}")
        print(f"Cref_max : {Cref_max}")
        print(f"Dref_min: {Dref_min}")
        print(f"Dref_max : {Dref_max}")

        norm_makespan=(makespan-Cref_min)/(Cref_max-Cref_min+1e-5)
        norm_load_imb=(max(loads)-min(loads)-Dref_min)/(Dref_max-Dref_min+1e-5)        
        objective=alpha* norm_makespan+beta* norm_load_imb
        
        print(f"Normalized Makespan: {norm_makespan:.4f}")
        print(f"Normalized Load Imbalance: {norm_load_imb:.4f}")
        print(f"Objective Function Value: {objective:.4f}")
        return assignment,schedule,makespan,loads


if __name__=='__main__':
    main()