# Airport Gate Assignment Problem (AGAP)

## Problem Definition

The Airport Gate Assignment Problem (AGAP) consists of assigning arriving and departing flights to airport gates while respecting operational constraints. This is a classical NP-hard combinatorial optimization problem in operations research.

**Constraints:**
- Each flight occupies exactly one gate (or apron/remote stand)
- No two overlapping flights can share the same gate
- Cleaning time between consecutive flights at the same gate
- Size compatibility

## Mathematical Formulation

**Sets:**
- $F$: set of flights, $|F| = n$
- $G$: set of gates, $|G| = m$ (gate 0 = apron)

**Parameters:**
- $a_f$: arrival time of flight $f$ (minutes since midnight)
- $d_f$: departure time of flight $f$
- $s_f \in \{1,2,3\}$: size of flight (1=small, 2=medium, 3=large)
- $c_g \in \{1,2,3\}$: capacity of gate $g$
- $t_{\text{clean}}$: cleaning time between flights (minutes)
- $M = 100 \times n$: penalty for apron assignment

**Decision Variables:**
$$x_{f,g} = \begin{cases} 1 & \text{if flight } f \text{ assigned to gate } g \\ 0 & \text{otherwise} \end{cases}$$

**Objective:**
$$max \sum_{f \in F} \sum_{g \in G, g \neq 0} x_{f,g} - M \sum_{f \in F} x_{f,0}$$

**Constraints:**

Each flight to exactly one gate:
$$\sum_{g \in G} x_{f,g} = 1 \quad \forall f \in F$$

No overlapping flights on same gate (with cleaning time):
$$x_{i,g} + x_{j,g} \leq 1 \quad \forall g \in G \setminus \{0\}, \forall i,j: a_i < d_j + t_{\text{clean}} \text{ and } a_j < d_i + t_{\text{clean}}$$

Size compatibility:
$$x_{f,g} \leq \text{compat}_{f,g} \quad \text{where } \text{compat}_{f,g} = 1 \text{ if } g=0 \text{ or } s_f \leq c_g$$

## Solution Methods

### MILP (Mixed Integer Linear Programming)

The exact formulation is solved using the CBC solver via PuLP. Guarantees optimality for small instances ($n \leq 20$). Computational complexity grows exponentially with problem size.

### Simulated Annealing (SA)

A metaheuristic inspired by the annealing process in metallurgy. Accepts worse solutions with decreasing probability to escape local optima. Provides near-optimal solutions faster than MILP for larger instances.

**SA Parameters:**
- Initial temperature: 100
- Cooling rate: 0.95
- Iterations per temperature: $100 \times n_{\text{flights}}$
- Stopping temperature: 0.01

## Results

For a test instance with 15 flights and 5 gates (2 small, 2 medium, 1 large):
- Optimal objective value: 14 (one large flight assigned to apron)
- Both MILP and SA achieve the same objective
- Multiple optimal assignments exist due to gate symmetry

## References

1. Bouras, A., Ghaleb, M. A., Suryahatmoko, U. S., & Hamdan, S. B. (2014). The airport gate assignment problem: A survey. *Journal of Air Transport Management*, 42, 1-13.

2. Dorndorf, U., Drexl, A., Nikulin, Y., & Pesch, E. (2007). Flight gate scheduling: State-of-the-art and recent developments. *Omega*, 35(3), 326-334.

3. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.

4. Wolsey, L. A. (2020). *Integer Programming*. John Wiley & Sons.
