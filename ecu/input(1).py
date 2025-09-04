import random
from collections import defaultdict, deque

# Resource Dictionary
RESOURCE_DICT = {
    "sensors": ["R1", "R1","R1", "R1", "R2", "R2", "R2", "R2", "R3", "R3", "R3", "R3", "R3", "R3", "R3", "R3", "R3", "R3"],
    "cpus": ["R4", "R4", "R4", "R4","R4", "R4", "R4", "R4", "R4", "R4", "R5", "R5", "R5", "R5", "R5", "R5", "R5", "R5", "R5", "R5"], 
    "actuators": ["R6", "R6", "R6","R6", "R6", "R6", "R6", "R6", "R6","R6","R7", "R7", "R7", "R7", "R7","R7", "R7", "R7","R7", "R7"]
}

ECU = [
    {"id": "E1", "cores": 2, "resources": ["R1", "R4", "R5", "R6", "R7"]},
    {"id": "E2", "cores": 2, "resources": ["R1", "R4", "R5", "R6", "R7"]},
    {"id": "E3", "cores": 2, "resources": ["R2", "R3", "R4", "R5", "R6", "R7"]},
    {"id": "E4", "cores": 2, "resources": ["R2", "R3", "R4", "R5", "R6", "R7"]},
    {"id": "E5", "cores": 2, "resources": ["R1", "R3", "R4", "R5", "R6", "R7"]},
    {"id": "E6", "cores": 2, "resources": ["R1", "R3", "R4", "R5", "R6", "R7"]},
    {"id": "E7", "cores": 2, "resources": ["R3", "R3", "R4", "R5", "R6", "R7"]},
    {"id": "E8", "cores": 2, "resources": ["R2", "R3", "R4", "R5", "R6", "R7"]},
    {"id": "E9", "cores": 2, "resources": ["R2", "R3", "R4", "R5", "R6", "R7"]},
    {"id": "E10", "cores": 2, "resources": ["R3", "R3", "R4", "R5", "R6", "R7"]}


]


def resource_generate(subtask_type=None, used_sensors=None):
    resources = []
    if subtask_type == "sensor":
        available_sensors = list(set(RESOURCE_DICT["sensors"]) - set(used_sensors or []))
        if available_sensors:
            resources.append(random.choice(available_sensors))
        else:
            resources.append(random.choice(RESOURCE_DICT["sensors"]))
    elif subtask_type == "compute":
        resources.append(random.choice(RESOURCE_DICT["cpus"]))
        if random.random() < 0.5:
            resources.append(random.choice(RESOURCE_DICT["sensors"]))
    elif subtask_type == "actuator":
        resources.append(random.choice(RESOURCE_DICT["actuators"]))

    if not resources:
        all_resources = RESOURCE_DICT["sensors"] + RESOURCE_DICT["cpus"] + RESOURCE_DICT["actuators"]
        resources.append(random.choice(all_resources))

    return list(set(resources))


def subtask_generate(subtask_id, subtask_type=None, used_sensors=None):
    operation = f"O{subtask_id}"
    t = random.randint(5, 10)
    b = random.randint(1, 5)
    n_mul = random.randint(1, 5)
    n = b * n_mul
    x_mul = random.randint(1, 5)
    x = n * x_mul
    resources = resource_generate(subtask_type, used_sensors)
    if used_sensors is not None:
        for r in resources:
            if r in RESOURCE_DICT["sensors"]:
                used_sensors.add(r)
    return {
        "id": subtask_id,
        "operation": operation,
        "t": t,
        "n": n,
        "b": b,
        "x": x,
        "resources": resources,
        "type": subtask_type
    }


def enforce_resource_precedence(dag):
    sensor_nodes = [st["id"] for st in dag["subtasks"] if st["type"] == "sensor"]
    compute_nodes = [st["id"] for st in dag["subtasks"] if st["type"] == "compute"]
    actuator_nodes = [st["id"] for st in dag["subtasks"] if st["type"] == "actuator"]
    edges = set()
    for c in compute_nodes:
        sources = [s for s in sensor_nodes if s != c]
        if sources:
            edges.add((random.choice(sources), c))
    for a in actuator_nodes:
        sources = [c for c in compute_nodes if c != a]
        if sources:
            edges.add((random.choice(sources), a))
    while not is_connected(len(dag["subtasks"]), list(edges)):
        u, v = random.sample([st["id"] for st in dag["subtasks"]], 2)
        if u != v and (u, v) not in edges:
            edges.add((u, v))
            if topological_sort(len(dag["subtasks"]), list(edges)) is None:
                edges.remove((u, v))
    return list(edges)


def is_connected(num_nodes, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    visited = set()

    def dfs(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)

    dfs(1)
    return len(visited) == num_nodes


def topological_sort(num_nodes, edges):
    adj = defaultdict(list)
    in_degree = [0] * (num_nodes + 1)
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    queue = deque([i for i in range(1, num_nodes + 1) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return topo_order if len(topo_order) == num_nodes else None


def ensure_single_actuator_output(dag):
    subtasks = dag["subtasks"]
    num_nodes = len(subtasks)
    output_node = random.choice(subtasks)
    output_node_id = output_node["id"]
    output_node["type"] = "actuator"
    output_node["resources"] = [random.choice(RESOURCE_DICT["actuators"])]

    for st in subtasks:
        if st["id"] != output_node_id and st["type"] == "actuator":
            st["type"] = "compute"
            st["resources"] = resource_generate("compute")

    dag["precedence"] = [(u, v) for (u, v) in dag["precedence"] if u != output_node_id]

    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    for u, v in dag["precedence"]:
        graph[u].append(v)
        reverse_graph[v].append(u)

    all_ids = set(st["id"] for st in subtasks)
    outgoing_srcs = set(u for u, v in dag["precedence"])
    nodes_with_no_outgoing = all_ids - outgoing_srcs

    def is_reachable(start, goal):
        visited = set()
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == goal:
                return True
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return False

    for node_id in nodes_with_no_outgoing:
        if node_id != output_node_id:
            if not reverse_graph[node_id]:
                continue
            if is_reachable(node_id, output_node_id):
                continue
            dag["precedence"].append((node_id, output_node_id))
            graph[node_id].append(output_node_id)

    dag["precedence"] = [(u, v) for (u, v) in dag["precedence"] if u != v]
    return dag


def dag_generate(dag_id, num_subtasks):
    num_sensor = max(1, num_subtasks // 3)
    num_actuator = 1
    num_compute = num_subtasks - num_sensor - num_actuator
    types = ["sensor"] * num_sensor + ["compute"] * num_compute + ["actuator"] * num_actuator
    random.shuffle(types)
    subtasks = []
    used_sensors = set()
    for i, stype in enumerate(types):
        subtasks.append(subtask_generate(i + 1, stype, used_sensors))
    dag = {
        "dag_id": dag_id,
        "subtasks": subtasks,
        "precedence": []
    }
    dag["precedence"] = enforce_resource_precedence(dag)
    dag = ensure_single_actuator_output(dag)
    return dag


def validate_dag(dag):
    return topological_sort(len(dag["subtasks"]), dag["precedence"]) is not None and is_connected(len(dag["subtasks"]),
                                                                                                  dag["precedence"])


def main():
    num_dags = int(input("Number of DAGs per test case: "))
    num_test_cases = int(input("Number of test cases: "))
    with open("input.txt", "w") as f:
        f.write(f"{num_test_cases}\n")
        for tc in range(num_test_cases):
            f.write(f"TestCase {tc + 1}\n{num_dags}\n")
            dags = []
            for i in range(num_dags):
                while True:
                    num_subtasks = random.randint(4, 6)
                    dag = dag_generate(i + 1, num_subtasks)
                    if validate_dag(dag):
                        dags.append(dag)
                        break
            for dag in dags:
                f.write(f"DAG {dag['dag_id']} {len(dag['subtasks'])}\n")
                for st in dag["subtasks"]:
                    res_str = ",".join(st["resources"])
                    f.write(f"{st['operation']} {st['t']} {st['n']} {st['x']} {st['b']} {res_str}\n")
                f.write(f"{len(dag['precedence'])}\n")
                for u, v in dag["precedence"]:
                    f.write(f"{u} {v}\n")

            # ECU output: no core info, include sensor/cpu/actuator count
            f.write(f"{len(ECU)}\n")
            for ecu in ECU:
                sensor_count = sum(1 for r in ecu["resources"] if r in RESOURCE_DICT["sensors"])
                cpu_count = sum(1 for r in ecu["resources"] if r in RESOURCE_DICT["cpus"])
                actuator_count = sum(1 for r in ecu["resources"] if r in RESOURCE_DICT["actuators"])
                res_str = ",".join(ecu["resources"])
                f.write(f"{ecu['id']} {sensor_count} {cpu_count} {actuator_count} {res_str}\n")


if __name__ == "__main__":
    main()
