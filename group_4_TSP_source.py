import numpy as np
import random
import statistics
import tracemalloc
import time


def load_text_file(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            # Split the line by spaces and convert elements to integers or floats
            elements = [float(element) for element in line.strip().split()]
            adjacency_matrix.append(elements)
    return np.array(adjacency_matrix)

def greedy_least_path(adjacency_matrix):
    num_nodes = adjacency_matrix.shape[0]
    path = []

    # Start from the first node (row 0)
    current_node = 0
    visited = set()
    visited.add(current_node)
    total_cost = 0

    while len(visited) < num_nodes:
        path.append(current_node)
        min_distance = np.inf
        next_node = None

        # Find the closest unvisited node
        for neighbor in range(num_nodes):
            if neighbor not in visited and 0 <= adjacency_matrix[current_node][neighbor] < min_distance:
                next_node = neighbor
                min_distance = adjacency_matrix[current_node][neighbor]

        # If there's a valid next node, move to it
        if next_node is not None:
            total_cost += min_distance
            current_node = next_node
            visited.add(current_node)
        else:
            # If no unvisited nodes are left, break the loop
            break

    # Add the last node to complete the path
    path.append(current_node)
    return total_cost


def BnB_DFS(adjacency_matrix, upper_bound):
    num_nodes = adjacency_matrix.shape[0]
    best_path = None
    best_weight = np.inf
    pruned_branches = {'count': 0}

    def cost(p):
        return sum(adjacency_matrix[p[i]][p[i + 1]] for i in range(len(p) - 1))

    def branch(partial_path):
        nonlocal best_path, best_weight
        if len(partial_path) == num_nodes:
            partial_weight = cost(partial_path)
            if partial_weight < best_weight:
                best_path = partial_path[:]
                best_weight = partial_weight
        else:
            last_node = partial_path[-1]
            for next_node in range(num_nodes):
                if next_node not in partial_path:
                    extended_path = partial_path + [next_node]
                    extended_weight = cost(extended_path)
                    if extended_weight < best_weight and extended_weight <= upper_bound:
                        branch(extended_path)
                    else:
                        pruned_branches['count'] += 1

    # Start the search from each node as the beginning of the path
    for start_node in range(num_nodes):
        branch([start_node])

    #make the graph a cycle and add start node
    best_weight += adjacency_matrix[best_path[0]][len(best_path)-1]
    best_path.append(best_path[0])

    total_tour_cost = calculate_tour_cost(adjacency_matrix, best_path)

    return best_path, best_weight, total_tour_cost, pruned_branches

# PATH COST CALCULATOR
def calculate_tour_cost(nodeNetwork, tour):
    n = len(tour)
    tour_total = 0

    for i in range(1, n):
        tour_total += nodeNetwork[tour[i - 1], tour[i]]

    tour_total += nodeNetwork[tour[0], tour[-1]]
    return tour_total

def calculate_longest_step(nodeNetwork, tour):
    n = len(tour)
    tour_total = 0
    next_step = 0
    highest_cost = 0
    most_expensive_index = 0

    for i in range(1, n):
        next_step = nodeNetwork[tour[i - 1], tour[i]]
        if next_step > highest_cost:
          highest_cost = next_step
          most_expensive_index = i
        #tour_total += next_step

    tour_total += nodeNetwork[tour[0], tour[-1]]
    return most_expensive_index

def SLS(adjacency_matrix):
    ## Variance calculation
    variances = [np.inf]
    n = len(adjacency_matrix)
    longest_step = 0
    lowvar_node = np.argmin(variances)
    slsNodes = list(range(1, n))
    slsNetwork = adjacency_matrix.copy()

    i = 1
    while i < n:
        variances.append(statistics.variance(slsNetwork[:,i]))
        i += 1

    ## first step
    tours_tried = []
    random.shuffle(slsNodes)
    tours_tried.append(slsNodes.copy())
    lowest_cost = calculate_tour_cost(adjacency_matrix,slsNodes)
    best_tour = slsNodes.copy()


    longest_step = calculate_longest_step(slsNetwork, slsNodes)
    slsNodes[lowvar_node], slsNodes[longest_step] = slsNodes[longest_step], slsNodes[lowvar_node]

    i = 0
    total_iterations = 0

    while i < 9000:
        inner_iteration = 0
        while lowest_cost > calculate_tour_cost(adjacency_matrix,slsNodes):
            lowest_cost = calculate_tour_cost(adjacency_matrix,slsNodes)
            tours_tried.append(slsNodes.copy())
            best_tour = slsNodes.copy()
            longest_step = calculate_longest_step(slsNetwork, slsNodes)
            slsNodes[lowvar_node], slsNodes[longest_step] = slsNodes[longest_step], slsNodes[lowvar_node]
            inner_iteration += 1
            total_iterations += 1
        random.shuffle(slsNodes)
        i += 1
    best_tour.insert(0,0)
    best_tour.append(0)

    total_tour_cost = calculate_tour_cost(adjacency_matrix, best_tour)


    return best_tour, lowest_cost, total_tour_cost, total_iterations


#Load Matrix
adj_matrix = load_text_file('15_20.0_5.0.out')

#start tracking time and memory usage

tracemalloc.start()
start_time = time.time()
#run greedy upper bound heuristic
heuristic = greedy_least_path(adj_matrix)
BnB_optimal_path, BnB_optimal_weight, BnB_tour_cost, BnB_pruned_branches = BnB_DFS(adj_matrix, heuristic)
print("BnB Max Mem Used: ", tracemalloc.get_traced_memory()[1])
print("BnB Time: %s seconds" % (time.time() - start_time))
print("BnB Optimal Path:", BnB_optimal_path)
print("BnB Optimal Weight:", BnB_optimal_weight)
print("BnB Total Tour Cost:", BnB_tour_cost)
print("BnB Number of Pruned Branches:", BnB_pruned_branches['count'])

tracemalloc.stop()
#end memory tracking

print("\n")
#start tracking memory usage
tracemalloc.start()
start_time = time.time()
#run SLS
SLS_optimal_path, SLS_optimal_weight, SLS_tour_cost, SLS_num_iterations = SLS(adj_matrix)
print("SLS Max Mem Used: ", tracemalloc.get_traced_memory()[1])
print("SLS Time: %s seconds" % (time.time() - start_time))
print("SLS Optimal Path:", SLS_optimal_path)
print("SLS Optimal Weight:", SLS_optimal_weight)
print("SLS Total Tour Cost:", SLS_tour_cost)
print("SLS Number of Iterations:", SLS_num_iterations)

tracemalloc.stop()
#end memory tracking