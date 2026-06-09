import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pulp
import random
import math
import copy
from collections import deque
import time

def generate_scenario(n_flights=15, n_gates=5, seed=42, t_clean=30):
    """Generate realistic airport scenario"""
    random.seed(seed)
    np.random.seed(seed)
    
    if n_gates == 1:
        gate_capacities = [2]
    elif n_gates == 2:
        gate_capacities = [1, 2]
    elif n_gates == 3:
        gate_capacities = [1, 2, 3]
    elif n_gates == 4:
        gate_capacities = [1, 1, 2, 3]
    elif n_gates == 5:
        gate_capacities = [1, 1, 2, 2, 3]
    elif n_gates == 6:
        gate_capacities = [1, 1, 2, 2, 3, 3]
    elif n_gates == 7:
        gate_capacities = [1, 1, 2, 2, 2, 3, 3]
    elif n_gates == 8:
        gate_capacities = [1, 1, 1, 2, 2, 2, 3, 3]
    elif n_gates == 9:
        gate_capacities = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    else:
        n_small = int(n_gates * 0.4)
        n_medium = int(n_gates * 0.4)
        n_large = n_gates - n_small - n_medium
        gate_capacities = [1] * n_small + [2] * n_medium + [3] * n_large
        random.shuffle(gate_capacities)
    
    gates = []
    for g, cap in enumerate(gate_capacities, start=1):
        gates.append({'id': g, 'capacity': cap})
    
    flights = []
    sizes = [1, 2, 3]
    size_weights = [0.4, 0.4, 0.2]
    
    start_min = 360  
    end_min = 1320   
    min_duration = 45
    max_duration = 180
    
    for f in range(n_flights):
        size = random.choices(sizes, weights=size_weights)[0]
        duration = random.randint(min_duration, max_duration)
        latest_arrival = end_min - duration
        if latest_arrival < start_min:
            latest_arrival = start_min
        arrival = random.randint(start_min, latest_arrival)
        departure = arrival + duration
        
        flights.append({
            'id': f,
            'arrival': arrival,
            'departure': departure,
            'size': size,
            'duration': duration
        })
    
    flights.sort(key=lambda x: x['arrival'])
    for idx, f in enumerate(flights):
        f['id'] = idx
    
    conflict_matrix = [[0 for _ in range(n_flights)] for _ in range(n_flights)]
    for i in range(n_flights):
        for j in range(i+1, n_flights):
            f1 = flights[i]
            f2 = flights[j]
            overlap = (f1['arrival'] < f2['departure'] + t_clean) and (f2['arrival'] < f1['departure'] + t_clean)
            if overlap:
                conflict_matrix[i][j] = 1
                conflict_matrix[j][i] = 1
    
    return {
        'flights': flights,
        'gates': gates,
        'apron_id': 0,
        't_clean': t_clean,
        'conflict_matrix': conflict_matrix,
        'n_flights': n_flights,
        'n_gates': n_gates
    }

def load_scenario_from_csv(data_path):
    flights_df = pd.read_csv(os.path.join(data_path, 'flights_with_times.csv'))
    flights = []
    for _, row in flights_df.iterrows():
        flights.append({
            'id': int(row['id']),
            'arrival': int(row['arrival']),
            'departure': int(row['departure']),
            'size': int(row['size']),
            'duration': int(row['duration'])
        })
    
    gates_df = pd.read_csv(os.path.join(data_path, 'gates_with_capacity.csv'))
    gates = []
    for _, row in gates_df.iterrows():
        gates.append({
            'id': int(row['id']),
            'capacity': int(row['capacity'])
        })
    
    conflict_matrix = pd.read_csv(os.path.join(data_path, 'conflict_matrix.csv'), header=None).values.tolist()
    
    return {
        'flights': flights,
        'gates': gates,
        'apron_id': 0,
        't_clean': 30,
        'conflict_matrix': conflict_matrix,
        'n_flights': len(flights),
        'n_gates': len(gates)
    }

def check_feasibility(solution, scenario):
    """Check if a solution is feasible - shared by GA and TS"""
    n_flights = scenario['n_flights']
    n_gates = scenario['n_gates']
    conflict_matrix = scenario['conflict_matrix']
    flights = scenario['flights']
    gates = scenario['gates']
    apron_id = scenario['apron_id']
    
    gate_flights = {g: [] for g in range(n_gates + 1)}
    for f, gate in enumerate(solution):
        gate_flights[gate].append(f)
    
    for gate in range(1, n_gates + 1):
        flights_at_gate = gate_flights[gate]
        for i in range(len(flights_at_gate)):
            for j in range(i + 1, len(flights_at_gate)):
                if conflict_matrix[flights_at_gate[i]][flights_at_gate[j]] == 1:
                    return False
    
    for f, gate in enumerate(solution):
        if gate != apron_id:
            flight_size = flights[f]['size']
            gate_capacity = gates[gate - 1]['capacity']
            if flight_size > gate_capacity:
                return False
    return True

def count_gate_assignments(solution, apron_id):
    """Count flights assigned to gates (not apron)"""
    return sum(1 for gate in solution if gate != apron_id)

def milp_style_objective(solution, scenario):
    """Convert solution to MILP-compatible objective value"""
    apron_id = scenario['apron_id']
    n_flights = scenario['n_flights']
    gate_count = count_gate_assignments(solution, apron_id)
    apron_count = n_flights - gate_count
    M = 100 * n_flights
    return gate_count - (M * apron_count)

def get_compatible_gates(flight_id, scenario):
    """Return list of gates (including apron) compatible with flight size"""
    apron_id = scenario['apron_id']
    flights = scenario['flights']
    gates = scenario['gates']
    
    compatible = [apron_id]
    flight_size = flights[flight_id]['size']
    for g in range(1, scenario['n_gates'] + 1):
        gate_capacity = gates[g - 1]['capacity']
        if flight_size <= gate_capacity:
            compatible.append(g)
    return compatible

def generate_random_feasible_solution(scenario):
    """Generate diverse feasible solutions"""
    apron_id = scenario['apron_id']
    n_flights = scenario['n_flights']
    
    for attempt in range(100):
        solution = [apron_id] * n_flights
        flight_order = list(range(n_flights))
        random.shuffle(flight_order)
        
        for f in flight_order:
            compatible = [g for g in get_compatible_gates(f, scenario) if g != apron_id]
            random.shuffle(compatible)
            
            for gate in compatible:
                conflict = False
                for other_f in range(n_flights):
                    if solution[other_f] == gate and scenario['conflict_matrix'][f][other_f]:
                        conflict = True
                        break
                if not conflict:
                    solution[f] = gate
                    break
        
        if check_feasibility(solution, scenario):
            return solution
    
    return [apron_id] * n_flights

class GeneticAlgorithmAGAP:
    def __init__(self, scenario, pop_size=100, generations=200, mutation_rate=0.1, crossover_rate=0.8):
        self.scenario = scenario
        self.n_flights = scenario['n_flights']
        self.n_gates = scenario['n_gates']
        self.apron_id = scenario['apron_id']
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.convergence_history = []
        
    def fitness(self, solution):
        """Death penalty fitness - returns gate count if feasible, -1e9 otherwise"""
        if not check_feasibility(solution, self.scenario):
            return -1000000
        return count_gate_assignments(solution, self.apron_id)
    
    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(self.n_flights):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        for child in [child1, child2]:
            for _ in range(10):
                if check_feasibility(child, self.scenario):
                    break
                for f in range(self.n_flights):
                    original = child[f]
                    compatible = get_compatible_gates(f, self.scenario)
                    random.shuffle(compatible)
                    for new_gate in compatible:
                        child[f] = new_gate
                        if check_feasibility(child, self.scenario):
                            break
                    if check_feasibility(child, self.scenario):
                        break
                    child[f] = original   
        
        return child1, child2
    
    def mutate(self, solution):
        """Mutate by reassigning a random flight to a compatible gate"""
        if random.random() > self.mutation_rate:
            return solution
        
        mutated = solution.copy()
        f = random.randint(0, self.n_flights - 1)
        compatible = get_compatible_gates(f, self.scenario)
        current_gate = solution[f]
        compatible = [g for g in compatible if g != current_gate]
        
        if compatible:
            new_gate = random.choice(compatible)
            mutated[f] = new_gate
            
            if not check_feasibility(mutated, self.scenario):
                mutated[f] = current_gate
        
        return mutated
    
    def select(self, population, fitnesses):
        """Tournament selection"""
        tournament_size = 3
        best_idx = max(random.sample(range(len(population)), tournament_size), 
                      key=lambda i: fitnesses[i])
        return population[best_idx].copy()
    
    def solve(self):
        population = []
        for _ in range(self.pop_size):
            population.append(generate_random_feasible_solution(self.scenario))
        
        best_solution = None
        best_fitness = -1e9
        
        for generation in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            
            gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
                self.convergence_history.append(best_fitness)
            else:
                self.convergence_history.append(best_fitness)
            
            new_population = []
            
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:2]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            while len(new_population) < self.pop_size:
                parent1 = self.select(population, fitnesses)
                parent2 = self.select(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            population = new_population
        
        return best_solution, self.convergence_history

def solve_agap_ga(scenario):
    """Genetic Algorithm for AGAP - maintains same output format"""
    ga = GeneticAlgorithmAGAP(scenario)
    best_solution, history = ga.solve()
    
    st.session_state['ga_history'] = history
    
    n_flights = scenario['n_flights']
    n_gates = scenario['n_gates']
    apron_id = scenario['apron_id']
    
    assignments = []
    apron_flights = []
    assignment_matrix = [[0 for _ in range(n_gates + 1)] for _ in range(n_flights)]
    
    for f, gate in enumerate(best_solution):
        assignment_matrix[f][gate] = 1
        if gate == apron_id:
            apron_flights.append(f)
        else:
            assignments.append((f, gate))
    
    return {
        'status': 'GA_Heuristic',
        'objective': milp_style_objective(best_solution, scenario),
        'assignments': assignments,
        'apron_flights': apron_flights,
        'assignment_matrix': assignment_matrix
    }

class TabuSearchAGAP:
    def __init__(self, scenario, max_iterations=500, tabu_tenure=10, neighborhood_size=30):
        self.scenario = scenario
        self.n_flights = scenario['n_flights']
        self.n_gates = scenario['n_gates']
        self.apron_id = scenario['apron_id']
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size
        self.convergence_history = []
        
    def fitness(self, solution):
        """Death penalty for TS as well"""
        if not check_feasibility(solution, self.scenario):
            return -1000000
        return count_gate_assignments(solution, self.apron_id)
    
    def get_neighbors(self, solution):
        neighbors = []
        
        for f in range(self.n_flights):
            current_gate = solution[f]
            compatible = get_compatible_gates(f, self.scenario)
            
            for new_gate in compatible:
                if new_gate == current_gate:
                    continue
                
                neighbor = solution.copy()
                neighbor[f] = new_gate
                
                if check_feasibility(neighbor, self.scenario):
                    fitness_val = self.fitness(neighbor)
                    neighbors.append((neighbor, fitness_val, f, new_gate))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:self.neighborhood_size]
    
    def solve(self):
        current = generate_random_feasible_solution(self.scenario)
        current_fitness = self.fitness(current)
        
        best = current.copy()
        best_fitness = current_fitness
        self.convergence_history.append(best_fitness)
        
        tabu_list = deque(maxlen=self.tabu_tenure)
        
        stagnation_counter = 0
        
        for iteration in range(self.max_iterations):
            neighbors = self.get_neighbors(current)
            
            if not neighbors:
                current = generate_random_feasible_solution(self.scenario)
                current_fitness = self.fitness(current)
                self.convergence_history.append(best_fitness)
                continue
            
            best_neighbor = None
            best_neighbor_fitness = -1e9
            best_f = None
            best_new_gate = None
            
            for neighbor, fitness_val, f, new_gate in neighbors:
                is_tabu = (f, new_gate) in tabu_list
                
                if fitness_val > best_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = fitness_val
                    best_f = f
                    best_new_gate = new_gate
                    break
                
                if not is_tabu and fitness_val > best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = fitness_val
                    best_f = f
                    best_new_gate = new_gate
            
            if best_neighbor is None:
                stagnation_counter += 1
                if stagnation_counter > 50:
                    current = generate_random_feasible_solution(self.scenario)
                    current_fitness = self.fitness(current)
                    stagnation_counter = 0
                    self.convergence_history.append(best_fitness)
                continue
            
            current = best_neighbor
            current_fitness = best_neighbor_fitness
            
            if best_f is not None and best_new_gate is not None:
                tabu_list.append((best_f, best_new_gate))
            
            if current_fitness > best_fitness:
                best = current.copy()
                best_fitness = current_fitness
                stagnation_counter = 0
            
            self.convergence_history.append(best_fitness)
        
        return best, self.convergence_history

def solve_agap_ts(scenario):
    """Tabu Search for AGAP - maintains same output format"""
    ts = TabuSearchAGAP(scenario)
    best_solution, history = ts.solve()
    
    st.session_state['ts_history'] = history
    
    n_flights = scenario['n_flights']
    n_gates = scenario['n_gates']
    apron_id = scenario['apron_id']
    
    assignments = []
    apron_flights = []
    assignment_matrix = [[0 for _ in range(n_gates + 1)] for _ in range(n_flights)]
    
    for f, gate in enumerate(best_solution):
        assignment_matrix[f][gate] = 1
        if gate == apron_id:
            apron_flights.append(f)
        else:
            assignments.append((f, gate))
    
    return {
        'status': 'TS_Heuristic',
        'objective': milp_style_objective(best_solution, scenario),
        'assignments': assignments,
        'apron_flights': apron_flights,
        'assignment_matrix': assignment_matrix
    }

def solve_agap_milp(scenario):
    n_flights = scenario['n_flights']
    n_gates = scenario['n_gates']
    gates = scenario['gates']
    flights = scenario['flights']
    conflict_matrix = scenario['conflict_matrix']
    apron_id = scenario['apron_id']
    
    prob = pulp.LpProblem("AGAP_MILP", pulp.LpMaximize)
    
    x = {}
    for f in range(n_flights):
        for g in range(n_gates + 1):
            x[(f, g)] = pulp.LpVariable(f"x_{f}_{g}", lowBound=0, upBound=1, cat=pulp.LpBinary)
    
    M = 100 * n_flights
    objective = pulp.lpSum(x[(f, g)] for f in range(n_flights) for g in range(1, n_gates + 1))
    objective -= M * pulp.lpSum(x[(f, apron_id)] for f in range(n_flights))
    prob += objective
    
    for f in range(n_flights):
        prob += pulp.lpSum(x[(f, g)] for g in range(n_gates + 1)) == 1
    
    for g in range(1, n_gates + 1):
        for i in range(n_flights):
            for j in range(i + 1, n_flights):
                if conflict_matrix[i][j] == 1:
                    prob += x[(i, g)] + x[(j, g)] <= 1
    
    for f in range(n_flights):
        flight_size = flights[f]['size']
        for g in range(1, n_gates + 1):
            gate_capacity = gates[g-1]['capacity']
            if flight_size > gate_capacity:
                prob += x[(f, g)] == 0
    
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    prob.solve(solver)
    
    assignments = []
    apron_flights = []
    assignment_matrix = [[0 for _ in range(n_gates + 1)] for _ in range(n_flights)]
    
    for f in range(n_flights):
        for g in range(n_gates + 1):
            if pulp.value(x[(f, g)]) and pulp.value(x[(f, g)]) > 0.5:
                assignment_matrix[f][g] = 1
                if g == apron_id:
                    apron_flights.append(f)
                else:
                    assignments.append((f, g))
    
    return {
        'status': pulp.LpStatus[prob.status],
        'objective': pulp.value(prob.objective),
        'assignments': assignments,
        'apron_flights': apron_flights,
        'assignment_matrix': assignment_matrix
    }

class SimulatedAnnealingAGAP:
    def __init__(self, scenario, initial_temp=100, cooling_rate=0.95, iterations_per_temp=100, min_temp=0.01):
        random.seed(42)
        self.scenario = scenario
        self.n_flights = scenario['n_flights']
        self.n_gates = scenario['n_gates']
        self.gates = scenario['gates']
        self.flights = scenario['flights']
        self.conflict_matrix = scenario['conflict_matrix']
        self.apron_id = scenario['apron_id']
        self.M = 100 * self.n_flights
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.min_temp = min_temp
    
    def is_feasible(self, solution):
        gate_flights = {g: [] for g in range(self.n_gates + 1)}
        for f, gate in enumerate(solution):
            gate_flights[gate].append(f)
        
        for gate in range(1, self.n_gates + 1):
            flights_at_gate = gate_flights[gate]
            for i in range(len(flights_at_gate)):
                for j in range(i + 1, len(flights_at_gate)):
                    if self.conflict_matrix[flights_at_gate[i]][flights_at_gate[j]] == 1:
                        return False
        
        for f, gate in enumerate(solution):
            if gate != self.apron_id:
                flight_size = self.flights[f]['size']
                gate_capacity = self.gates[gate - 1]['capacity']
                if flight_size > gate_capacity:
                    return False
        return True
    
    def calculate_objective(self, solution):
        real_gate_count = sum(1 for gate in solution if gate != self.apron_id)
        apron_count = sum(1 for gate in solution if gate == self.apron_id)
        return real_gate_count - (self.M * apron_count)
    
    def get_compatible_gates(self, flight_id):
        compatible = [self.apron_id]
        flight_size = self.flights[flight_id]['size']
        for g in range(1, self.n_gates + 1):
            gate_capacity = self.gates[g - 1]['capacity']
            if flight_size <= gate_capacity:
                compatible.append(g)
        return compatible
    
    def generate_initial_solution(self):
        solution = [self.apron_id] * self.n_flights
        for f in range(self.n_flights):
            compatible_gates = [g for g in self.get_compatible_gates(f) if g != self.apron_id]
            for gate in compatible_gates:
                conflict = False
                for other_f in range(self.n_flights):
                    if solution[other_f] == gate and self.conflict_matrix[f][other_f] == 1:
                        conflict = True
                        break
                if not conflict:
                    solution[f] = gate
                    break
        return solution
    
    def get_neighbor(self, solution):
        for attempt in range(100):
            neighbor = copy.deepcopy(solution)
            f = random.randint(0, self.n_flights - 1)
            possible_gates = self.get_compatible_gates(f)
            current_gate = solution[f]
            possible_gates = [g for g in possible_gates if g != current_gate]
            if not possible_gates:
                continue
            random.shuffle(possible_gates)
            for new_gate in possible_gates:
                neighbor[f] = new_gate
                if self.is_feasible(neighbor):
                    return neighbor
        return solution
    
    def solve(self, verbose=False):
        current_solution = self.generate_initial_solution()
        current_objective = self.calculate_objective(current_solution)
        best_solution = copy.deepcopy(current_solution)
        best_objective = current_objective
        temperature = self.initial_temp
        
        while temperature > self.min_temp:
            for _ in range(self.iterations_per_temp):
                neighbor = self.get_neighbor(current_solution)
                neighbor_objective = self.calculate_objective(neighbor)
                if neighbor_objective > current_objective:
                    current_solution = neighbor
                    current_objective = neighbor_objective
                    if current_objective > best_objective:
                        best_solution = copy.deepcopy(current_solution)
                        best_objective = current_objective
                else:
                    delta = neighbor_objective - current_objective
                    if random.random() < math.exp(delta / temperature):
                        current_solution = neighbor
                        current_objective = neighbor_objective
            temperature *= self.cooling_rate
        
        return {'best_solution': best_solution, 'best_objective': best_objective}

def solve_agap_sa(scenario):
    sa = SimulatedAnnealingAGAP(scenario)
    result = sa.solve()
    
    n_flights = scenario['n_flights']
    n_gates = scenario['n_gates']
    
    assignments = []
    apron_flights = []
    assignment_matrix = [[0 for _ in range(n_gates + 1)] for _ in range(n_flights)]
    
    for f, gate in enumerate(result['best_solution']):
        assignment_matrix[f][gate] = 1
        if gate == scenario['apron_id']:
            apron_flights.append(f)
        else:
            assignments.append((f, gate))
    
    return {
        'status': 'SA_Heuristic',
        'objective': result['best_objective'],
        'assignments': assignments,
        'apron_flights': apron_flights,
        'assignment_matrix': assignment_matrix
    }

st.set_page_config(page_title="AGAP Dashboard", layout="wide")

st.title("✈️ Airport Gate Assignment Problem")
st.markdown("MILP vs Simulated Annealing vs Genetic Algorithm vs Tabu Search - Generate scenario or load from CSV")

st.sidebar.header("📂 Data Source")

data_source = st.sidebar.radio(
    "Select data source",
    ["Generate New Scenario", "Load Existing CSV"]
)

scenario = None

if data_source == "Generate New Scenario":
    st.sidebar.subheader("Scenario Parameters")
    
    n_flights = st.sidebar.slider("Number of Flights", min_value=8, max_value=30, value=15, 
                                   help="Minimum 8 flights (problem too trivial below). Maximum 30 (performance limit for MILP).")
    n_gates = st.sidebar.slider("Number of Gates", min_value=3, max_value=10, value=5,
                                 help="Minimum 3 gates (small/medium/large diversity). Maximum 10 (UI readability).")
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=999)
    t_clean = st.sidebar.slider("Cleaning Time (minutes)", min_value=15, max_value=60, value=30)
    
    ratio = n_flights / n_gates
    if ratio < 2:
        st.sidebar.warning(f"⚠️ Low flight/gate ratio ({ratio:.1f}:1). Problem may be too easy. Consider more flights.")
    elif ratio > 6:
        st.sidebar.warning(f"⚠️ High flight/gate ratio ({ratio:.1f}:1). MILP may be slow. Consider fewer flights.")
    else:
        st.sidebar.success(f"✅ Good ratio: {ratio:.1f} flights per gate")
    
    if st.sidebar.button("Generate Scenario", type="primary"):
        with st.spinner("Generating scenario..."):
            scenario = generate_scenario(n_flights, n_gates, seed, t_clean)
            st.session_state['scenario'] = scenario
            st.session_state['scenario_source'] = "generated"
            st.success(f"✅ Generated: {n_flights} flights, {n_gates} gates")
            st.rerun()

else:   
    st.sidebar.subheader("CSV Files Path")
    data_path = st.sidebar.text_input(
        "Dataset folder path",
        value=r"C:\Users\HessaM\Desktop\دیتاست\first_MILP_SA\airport gate optimization\dataset"
    )
    
    if st.sidebar.button("Load Dataset"):
        try:
            scenario = load_scenario_from_csv(data_path)
            st.session_state['scenario'] = scenario
            st.session_state['scenario_source'] = "loaded"
            st.success(f"✅ Loaded: {scenario['n_flights']} flights, {scenario['n_gates']} gates")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if 'scenario' in st.session_state:
    scenario = st.session_state['scenario']
    
    st.sidebar.header("⚙️ Algorithms")
    run_milp = st.sidebar.checkbox("MILP (Optimal)", value=True)
    run_sa = st.sidebar.checkbox("Simulated Annealing (Heuristic)", value=True)
    run_ga = st.sidebar.checkbox("Genetic Algorithm (Heuristic)", value=False)
    run_ts = st.sidebar.checkbox("Tabu Search (Heuristic)", value=False)
    
    if st.sidebar.button("🚀 Run Optimization", type="primary"):
        runtimes = {}
        
        if run_milp:
            with st.spinner("Solving with MILP (branch-and-bound)..."):
                start = time.time()
                milp_result = solve_agap_milp(scenario)
                runtimes['milp'] = time.time() - start
                st.session_state['milp_result'] = milp_result
                st.session_state['milp_runtime'] = runtimes['milp']
                st.success(f"MILP complete in {runtimes['milp']:.2f}s")
        
        if run_sa:
            with st.spinner("Solving with Simulated Annealing..."):
                start = time.time()
                sa_result = solve_agap_sa(scenario)
                runtimes['sa'] = time.time() - start
                st.session_state['sa_result'] = sa_result
                st.session_state['sa_runtime'] = runtimes['sa']
                st.success(f"SA complete in {runtimes['sa']:.2f}s")
        
        if run_ga:
            with st.spinner("Solving with Genetic Algorithm..."):
                start = time.time()
                ga_result = solve_agap_ga(scenario)
                runtimes['ga'] = time.time() - start
                st.session_state['ga_result'] = ga_result
                st.session_state['ga_runtime'] = runtimes['ga']
                st.success(f"GA complete in {runtimes['ga']:.2f}s")
        
        if run_ts:
            with st.spinner("Solving with Tabu Search..."):
                start = time.time()
                ts_result = solve_agap_ts(scenario)
                runtimes['ts'] = time.time() - start
                st.session_state['ts_result'] = ts_result
                st.session_state['ts_runtime'] = runtimes['ts']
                st.success(f"TS complete in {runtimes['ts']:.2f}s")
        
        st.rerun()
    
    st.subheader("📊 Scenario Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Flights", scenario['n_flights'])
    with col2:
        st.metric("Gates", scenario['n_gates'])
    with col3:
        st.metric("Flights/Gate Ratio", f"{scenario['n_flights']/scenario['n_gates']:.1f}")
    with col4:
        st.metric("Cleaning Time", f"{scenario['t_clean']} min")
    
    with st.expander("🚪 Gate Configuration"):
        gates_df = pd.DataFrame(scenario['gates'])
        gates_df['capacity_name'] = gates_df['capacity'].map({1:'Small',2:'Medium',3:'Large'})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Small Gates", sum(1 for g in scenario['gates'] if g['capacity'] == 1))
        with col2:
            st.metric("Medium Gates", sum(1 for g in scenario['gates'] if g['capacity'] == 2))
        with col3:
            st.metric("Large Gates", sum(1 for g in scenario['gates'] if g['capacity'] == 3))
        st.dataframe(gates_df[['id', 'capacity_name']])
    
    with st.expander("📋 Flight Schedule"):
        flights_df = pd.DataFrame(scenario['flights'])
        flights_df['arrival_time'] = flights_df['arrival'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")
        flights_df['departure_time'] = flights_df['departure'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")
        flights_df['size_name'] = flights_df['size'].map({1:'Small',2:'Medium',3:'Large'})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Small Flights", sum(1 for f in scenario['flights'] if f['size'] == 1))
        with col2:
            st.metric("Medium Flights", sum(1 for f in scenario['flights'] if f['size'] == 2))
        with col3:
            st.metric("Large Flights", sum(1 for f in scenario['flights'] if f['size'] == 3))
        st.dataframe(flights_df[['id', 'size_name', 'arrival_time', 'departure_time', 'duration']])
    
    results_exist = any(key in st.session_state for key in ['milp_result', 'sa_result', 'ga_result', 'ts_result'])
    
    if results_exist:
        st.subheader("📈 Results Comparison")
        
        available_results = []
        if 'milp_result' in st.session_state:
            available_results.append(('MILP', 'milp_result', '🟢'))
        if 'sa_result' in st.session_state:
            available_results.append(('SA', 'sa_result', '🟠'))
        if 'ga_result' in st.session_state:
            available_results.append(('GA', 'ga_result', '🔵'))
        if 'ts_result' in st.session_state:
            available_results.append(('TS', 'ts_result', '🟣'))
        
        cols = st.columns(len(available_results))
        
        for col, (name, key, color) in zip(cols, available_results):
            result = st.session_state[key]
            runtime = st.session_state.get(f'{name.lower()}_runtime', 0)
            with col:
                st.markdown(f"### {color} {name}")
                st.metric("Objective", f"{result['objective']:.0f}")
                st.metric("Assigned to Gates", len(result['assignments']))
                st.metric("Assigned to Apron", len(result['apron_flights']))
                st.metric("Runtime (s)", f"{runtime:.2f}")
                st.metric("Status", result['status'])
        
        st.subheader("📈 Algorithm Convergence")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'ga_history' in st.session_state and len(st.session_state['ga_history']) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(st.session_state['ga_history'])
                ax.set_xlabel('Generation')
                ax.set_ylabel('Best Gates Used')
                ax.set_title('GA Convergence')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("Run GA to see convergence plot")
        
        with col2:
            if 'ts_history' in st.session_state and len(st.session_state['ts_history']) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(st.session_state['ts_history'])
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Best Gates Used')
                ax.set_title('Tabu Search Convergence')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("Run TS to see convergence plot")
        
        st.subheader("⏱️ Performance Summary")
        perf_data = []
        for name, key, color in [('MILP','milp_result','🟢'), ('SA','sa_result','🟠'), 
                                  ('GA','ga_result','🔵'), ('TS','ts_result','🟣')]:
            if key in st.session_state:
                result = st.session_state[key]
                runtime = st.session_state.get(f'{name.lower()}_runtime', 0)
                gates_used = len(result['assignments'])
                perf_data.append({
                    'Algorithm': name,
                    'Gates Used': gates_used,
                    'Apron': len(result['apron_flights']),
                    'Runtime (s)': f"{runtime:.2f}",
                    'Gates/sec': f"{gates_used/max(runtime, 0.001):.1f}"
                })
        st.dataframe(pd.DataFrame(perf_data))
        
        st.subheader("📅 Gate Assignment Gantt Chart")
        
        algo_options = [name for name, _, _ in available_results]
        algo_option = st.radio("Select Algorithm to Visualize", algo_options, horizontal=True)
        
        result = None
        for name, key, _ in available_results:
            if algo_option == name:
                result = st.session_state[key]
                break
        
        if result:
            fig, ax = plt.subplots(figsize=(14, 6))
            colors = {1: '#90EE90', 2: '#87CEEB', 3: '#FFB6C1'}
            size_names = {1: 'S', 2: 'M', 3: 'L'}
            
            gate_flights = {g: [] for g in range(1, scenario['n_gates'] + 1)}
            for f, g in result['assignments']:
                gate_flights[g].append(f)
            
            y_pos = {}
            y = 0
            for g in range(1, scenario['n_gates'] + 1):
                y_pos[g] = y
                y += 1
            
            if result['apron_flights']:
                y_pos['apron'] = y
            
            for g, flights_at_gate in gate_flights.items():
                for f in flights_at_gate:
                    flight = scenario['flights'][f]
                    start = flight['arrival'] / 60
                    width = flight['duration'] / 60
                    ax.barh(y_pos[g], width, left=start, color=colors[flight['size']], edgecolor='black', linewidth=0.5)
                    ax.text(start + width/2, y_pos[g], f"{size_names[flight['size']]}{f}", ha='center', va='center', fontsize=9)
            
            if result['apron_flights']:
                for f in result['apron_flights']:
                    flight = scenario['flights'][f]
                    start = flight['arrival'] / 60
                    width = flight['duration'] / 60
                    ax.barh(y_pos['apron'], width, left=start, color='gray', edgecolor='black', linewidth=0.5, hatch='//', alpha=0.7)
                    ax.text(start + width/2, y_pos['apron'], f"A-{f}", ha='center', va='center', fontsize=9)
            
            ax.set_yticks(list(y_pos.values()))
            ax.set_yticklabels([f"Gate {g}" if g != 'apron' else "Apron" for g in y_pos.keys()])
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_title(f'{algo_option} - Gate Assignment', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(6, 22)
            
            small_patch = mpatches.Patch(color='#90EE90', label='Small')
            medium_patch = mpatches.Patch(color='#87CEEB', label='Medium')
            large_patch = mpatches.Patch(color='#FFB6C1', label='Large')
            apron_patch = mpatches.Patch(color='gray', hatch='//', label='Apron')
            ax.legend(handles=[small_patch, medium_patch, large_patch, apron_patch], loc='upper left')
            
            st.pyplot(fig)
            
            with st.expander("📋 Detailed Assignments"):
                data = []
                for f in range(scenario['n_flights']):
                    gate = None
                    for g in range(1, scenario['n_gates'] + 1):
                        if result['assignment_matrix'][f][g] == 1:
                            gate = g
                            break
                    if gate is None:
                        gate = "APRON" if f in result['apron_flights'] else "?"
                    
                    flight = scenario['flights'][f]
                    data.append({
                        'Flight': f,
                        'Size': {1:'S',2:'M',3:'L'}[flight['size']],
                        'Arrival': f"{flight['arrival']//60:02d}:{flight['arrival']%60:02d}",
                        'Departure': f"{flight['departure']//60:02d}:{flight['departure']%60:02d}",
                        'Gate': gate
                    })
                st.dataframe(pd.DataFrame(data))
            
            st.info("ℹ️ **Note:** Due to gate symmetry (identical gates with same capacity), multiple optimal assignments may exist. All algorithms aim to maximize gate utilization.")

st.markdown("---")
st.markdown("**AGAP Dashboard** | MILP (PuLP) vs Simulated Annealing vs Genetic Algorithm vs Tabu Search | Generate scenarios or load from CSV")
