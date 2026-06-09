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
st.markdown("MILP vs Simulated Annealing - Generate scenario or load from CSV")

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
        value=r"C:\Users\HessaM\Desktop\دیتاست\airport gate optimization\dataset"
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
    
    if st.sidebar.button("🚀 Run Optimization", type="primary"):
        if run_milp:
            with st.spinner("Solving with MILP (branch-and-bound)..."):
                milp_result = solve_agap_milp(scenario)
                st.session_state['milp_result'] = milp_result
                st.success("MILP complete")
        
        if run_sa:
            with st.spinner("Solving with Simulated Annealing..."):
                sa_result = solve_agap_sa(scenario)
                st.session_state['sa_result'] = sa_result
                st.success("SA complete")
        
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
    
    if 'milp_result' in st.session_state or 'sa_result' in st.session_state:
        st.subheader("📈 Results Comparison")
        
        col1, col2 = st.columns(2)
        
        if 'milp_result' in st.session_state:
            milp = st.session_state['milp_result']
            with col1:
                st.markdown("### 🟢 MILP (Exact)")
                st.metric("Objective", f"{milp['objective']:.0f}")
                st.metric("Assigned to Gates", len(milp['assignments']))
                st.metric("Assigned to Apron", len(milp['apron_flights']))
                st.metric("Status", milp['status'])
        
        if 'sa_result' in st.session_state:
            sa = st.session_state['sa_result']
            with col2:
                st.markdown("### 🟠 Simulated Annealing")
                st.metric("Objective", f"{sa['objective']:.0f}")
                st.metric("Assigned to Gates", len(sa['assignments']))
                st.metric("Assigned to Apron", len(sa['apron_flights']))
                st.metric("Status", sa['status'])
        
        st.subheader("📅 Gate Assignment Gantt Chart")
        
        algo_option = st.radio("Select Algorithm to Visualize", ["MILP", "Simulated Annealing"], horizontal=True)
        
        if algo_option == "MILP" and 'milp_result' in st.session_state:
            result = st.session_state['milp_result']
        elif algo_option == "Simulated Annealing" and 'sa_result' in st.session_state:
            result = st.session_state['sa_result']
        else:
            result = None
        
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
            
            st.info("ℹ️ **Note:** Due to gate symmetry (identical gates with same capacity), multiple optimal assignments may exist. Both algorithms achieve the same objective value.")

st.markdown("---")
st.markdown("**AGAP Dashboard** | MILP (PuLP) vs Simulated Annealing | Generate scenarios or load from CSV")
