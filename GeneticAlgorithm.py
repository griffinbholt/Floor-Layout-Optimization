import numpy as np
import pandas as pd
import pygad
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

NEARNESS_SCALE = {"A": 1, "E": 0, "I": 0, "O": 0, "U": 0, "X": 0}

def load_building_info(filename):
    W, L = pd.read_csv(filename).to_numpy()[0]
    return W, L

def load_cell_info(filename):
    cells = pd.read_csv(filename)
    names = cells["name"].to_numpy()
    min_areas = cells["min_area"].to_numpy()
    min_widths = cells["min_width"].to_numpy()
    return min_areas, min_widths, len(cells)

def load_relationship_graph(N, filename):
    relationships_df = pd.read_csv(filename, header=None)
    relationships_df[2] = relationships_df[2].map(NEARNESS_SCALE)
    relationships = relationships_df.to_numpy()

    must_be_close = []
    relationship_graph = nx.Graph()
    relationship_graph.add_nodes_from(range(N))
    for (i, j, weight) in relationships:
        if weight == NEARNESS_SCALE["A"]:
            must_be_close.append((i - 1, j - 1))
        relationship_graph.add_edge(i - 1, j - 1, weight=weight)
    
    return relationship_graph, must_be_close

W, L = load_building_info("./data/building.csv")
min_areas, min_widths, N = load_cell_info("./data/cells.csv")
min_areas = 0.25 * min_areas
max_area_weight = 1
relationship_graph, must_be_close = load_relationship_graph(N, "./data/relationship_chart.csv")
n_iter = 1

def main():
    def fitness(ga_instance, solution, solution_idx):
        return solve_optimal_layout(solution, N, W, L, min_areas, min_widths, max_area_weight, must_be_close)
    def on_generation(ga_instance):
        global n_iter
        print("Generation: ", n_iter)
        n_iter += 1

    initial_population = initialize_population(1000, N, relationship_graph)
    ga_instance = pygad.GA(num_generations=100,
                           num_parents_mating=100,
                           fitness_func=fitness,
                           initial_population=initial_population,
                           gene_type=int,
                           gene_space=[0, 1, 2, 3],
                           parent_selection_type="sss",
                           keep_parents=-1,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=5,
                           on_generation=on_generation)
    start_time = time.time()
    print("Start")
    ga_instance.run()
    finish_time = time.time()
    print("Seconds Elapsed: ", start_time - finish_time)

    best_chromosome, best_score, _ = ga_instance.best_solution()
    print(best_score)
    x, y, w, l = solve_optimal_layout(best_chromosome, N, W, L, min_areas, min_widths, max_area_weight, must_be_close, return_solution=True)
    plot_solution(N, x, y, w, l, W, L)

    ga_instance.plot_fitness()
    # ga_instance.plot_genes()
    # ga_instance.plot_new_solution_rate()

def initialize_population(sol_per_pop, N, relationship_graph):  # TODO - Randomly switch directions
    initial_population = []
    for _ in range(sol_per_pop):
        L, B = get_fruchterman_reingold_relative_positionings(relationship_graph)
        chromosome = np.random.choice([0, 2], size=int(N * (N - 1) / 2))
        k = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                if (chromosome[k] == 0 and L.has_edge(j, i)) or (chromosome[k] == 2 and B.has_edge(j, i)):
                    chromosome[k] += 1
        initial_population.append(chromosome)
    return initial_population

def get_fruchterman_reingold_relative_positionings(G):
    pos = nx.spring_layout(G)

    L = nx.DiGraph()
    L.add_nodes_from(G.nodes)
    B = L.copy()
    for i, j in G.edges():
        if pos[i][0] <= pos[j][0]:
            L.add_edge(i, j)
        else:
            L.add_edge(j, i)
        if pos[i][1] <= pos[j][1]:
            B.add_edge(i, j)
        else:
            B.add_edge(j, i)
    return L, B


def solve_optimal_layout(chromosome: list[int], 
                         N: int, 
                         W: float, 
                         L: float, 
                         min_areas: list[float], 
                         min_widths: list[float],
                         max_area_weight: float, 
                         must_be_close: list[tuple[int, int]],
                         return_solution: bool=False) -> float:
    """
    Basis for the fitness function for a specific chromosome.
  
    Scores a chromosome defining N(N - 1)/2 relative positions of the N rooms in the facility
    using CVXPY to find the optimal layout.
  
    Parameters:
    chromosome: list of integers {0, 1, 2, 3} of length N(N - 1)/2, defining the relationship between room i and j
                as "i left of j", "j left of i", "i below j", or "j below i", respectively
    N: the number of rooms in the facility
    W: the width of the building (horizontal)
    L: the length of the building (vertical)
    min_areas: list of floats of length N, defining the minimum required area of each room
    max_area_weight: a weight defining the importance of maximizing the areas of the rooms with respect to
                     the other objectives (minimizing distance between certain areas)
    must_be_close: TODO

  
    Returns:
    float: the score of the optimal layout under the given constraints (higher score = better)
    """
    x, y, w, l = cp.Variable(N), cp.Variable(N), cp.Variable(N), cp.Variable(N)

    objective_func = max_area_weight * (-cp.sum(cp.log(w)) - cp.sum(cp.log(l))) # Objective: Maximize Areas
    objective_func += cp.sum([cp.norm1(cp.vstack([x[i] + (w[i]/2) - x[j] - (w[j]/2), y[i] + (l[i]/2) - y[j] - (l[j]/2)])) for (i, j) in must_be_close])
    constraints = [x >= 0, y >= 0, w >= 5, l >= 5, x + w <= W, y + l <= L]      # Boundary Constraints

    constraints += [cp.log(w) + cp.log(l) >= np.log(min_areas)]                 # Minimum Area Constraints
    # constraints += [w[i] >= min_widths[i] for i in np.where(~np.isnan(min_widths))[0]]  # Minimum Widths     
    # constraints += [l[i] >= min_widths[i] for i in np.where(~np.isnan(min_widths))[0]]
    constraints += [w - 5*l <= 0, l - 5*w <= 0]  # Maximum & Minimum Ratio Constraints: 1/5 <= w/l <= 5

    k = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Relative Positioning Constraints
            if chromosome[k] == 0:
                constraints += [x[i] + w[i] <= x[j]]
            elif chromosome[k] == 1:
                constraints += [x[j] + w[j] <= x[i]]
            elif chromosome[k] == 2:
                constraints += [y[i] + l[i] <= y[j]]
            else:  # chromosome[k] == 3
                constraints += [y[j] + l[j] <= y[i]]
            k += 1
    
    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except cp.error.SolverError:
        print("Error")
        return -np.inf

    if problem.status == "infeasible":
        return -np.inf
    if return_solution:
        return x.value, y.value, w.value, l.value
    return -problem.value

def plot_solution(N, x, y, w, l, W, L):
    fig, ax = plt.subplots()
    plt.scatter(x, y, s=0.1, color="black")
    ax.add_patch(Rectangle((0, 0), W, L, edgecolor="black", linestyle="dashed", fill=False))
    for i in range(N):
        ax.add_patch(Rectangle((x[i], y[i]), w[i], l[i], edgecolor="tab:blue", fill=False))
        plt.text(x[i] + (w[i] / 2) - 1, y[i] + (l[i] / 2) - 1, s=str(i))
    # plt.savefig("./images/solution.pdf")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()