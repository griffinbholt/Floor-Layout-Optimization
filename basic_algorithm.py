import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
import pandas as pd

NEARNESS_SCALE = {"A": 100, "E": 50, "I": 25, "O": 10, "U": 5, "X": 0}

def main():
    W, L = load_building_info("./data/building.csv")
    min_areas, min_widths, N = load_cell_info("./data/cells.csv")
    relationship_graph, must_be_close, must_be_far = load_relationship_graph(N, "./data/relationship_chart.csv")
    Left, Below = get_relative_positionings(relationship_graph)
    fopt, x, y, w, l = find_optimal_layout(N, W, L, min_areas, min_widths, Left, Below, must_be_close, must_be_far)
    
    print("Optimal Objective: ", fopt)
    print("x:", x)
    print("y:", y)
    print("w:", w)
    print("l:", l)
    plot_solution(N, x, y, w, l, W, L)

def plot_solution(N, x, y, w, l, W, L):
    fig, ax = plt.subplots()
    plt.scatter(x, y, s=0.1, color="black")
    ax.add_patch(Rectangle((0, 0), W, L, edgecolor="black", linestyle="dashed", fill=False))
    for i in range(N):
        ax.add_patch(Rectangle((x[i], y[i]), w[i], l[i], edgecolor="tab:blue", fill=False))
        plt.text(x[i] + (w[i] / 2) - 1, y[i] + (l[i] / 2) - 1, s=str(i))
    plt.savefig("./images/solution.pdf")
    # plt.show()
    plt.close()

def find_optimal_layout(N, W, L, min_areas, min_widths, Left, Below, must_be_close, must_be_far):
    x = cp.Variable(N)
    y = cp.Variable(N)
    w = cp.Variable(N)
    l = cp.Variable(N)

    objective_func = cp.sum([cp.norm1(cp.vstack([x[i] + (w[i]/2) - x[j] - (w[j]/2), y[i] + (l[i]/2) - y[j] - (l[j]/2)])) for (i, j) in must_be_close])
    for (i, j) in must_be_far:
        if nx.has_path(Left, i, j):
            objective_func += x[i] + (w[i]/2) - x[j] - (w[j]/2)
        else:
            objective_func += x[j] + (w[j]/2) - x[i] - (w[i]/2)
        if nx.has_path(Below, i, j):
            objective_func += y[i] + (l[i]/2) - y[j] - (l[j]/2)
        else:
            objective_func += y[j] + (l[j]/2) - y[i] - (l[i]/2)
    objective_func += 10 * (- cp.sum(cp.log(w)) - cp.sum(cp.log(l)))
    objective = cp.Minimize(objective_func)

    constraints = [x >= 0, y >= 0, w >= 0, l >= 0, x + w <= W, y + l <= L, # Boundary Constraints
                   cp.log(w) + cp.log(l) >= np.log(min_areas/8)]  # Minimum Area Constraints              
    constraints += [x[i] + w[i] <= x[j] for (i, j) in Left.edges()]
    constraints += [y[i] + l[i] <= y[j] for (i, j) in Below.edges()]
    constraints += [w[i] >= min_widths[i] for i in np.where(~np.isnan(min_widths))[0]]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem.value, x.value, y.value, w.value, l.value
   
def get_relative_positionings(G, tol=0.1):
    pos = nx.spring_layout(G, seed=1)
    draw_graph(G, pos, filename="./images/spring_layout.pdf", draw_edge_labels=True)

    L = nx.DiGraph()
    L.add_nodes_from(G.nodes)
    B = L.copy()
    for i, j in G.edges():
        if tol <= pos[j][0] - pos[i][0]:
            L.add_edge(i, j)
        if tol <= pos[i][0] - pos[j][0]:
            L.add_edge(j, i)
        if tol <= pos[j][1] - pos[i][1]:
            B.add_edge(i, j)
        if tol <= pos[i][1] - pos[j][1]:
            B.add_edge(j, i)
    draw_graph(L, pos, filename="./images/left_rel_pos.pdf")
    draw_graph(B, pos, filename="./images/below_rel_pos.pdf")

    L = get_minimum_equivalent_digraph(L)
    B = get_minimum_equivalent_digraph(B)

    draw_graph(L, pos, filename="./images/left_rel_pos_min.pdf")
    draw_graph(B, pos, filename="./images/below_rel_pos_min.pdf")
    return L, B

def get_minimum_equivalent_digraph(G):
    N = G.number_of_nodes()
    D = G.copy()
    for j in range(N):
        for i in range(N):
            if D.has_edge(i, j):
                for k in D.neighbors(j):
                    if D.has_edge(i, k):
                        D.remove_edge(i, k)
    return D
    
    
def draw_graph(G, pos, filename=None, draw_edge_labels=False):
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos)
    if draw_edge_labels:
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, labels)
    if filename is not None:
        plt.savefig(filename)
    # plt.show()
    plt.close()

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
    must_be_far = []
    relationship_graph = nx.Graph()
    relationship_graph.add_nodes_from(range(N))
    for (i, j, weight) in relationships:
        if weight == NEARNESS_SCALE["A"]:
            must_be_close.append((i - 1, j - 1))
        if weight == NEARNESS_SCALE["X"]:
            must_be_far.append((i - 1, j - 1))
        relationship_graph.add_edge(i - 1, j - 1, weight=weight)
    
    return relationship_graph, must_be_close, must_be_far

if __name__ == "__main__":
    main()