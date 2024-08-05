# **Bidirectional Traveling Salesman Problem(TSP)**
- Shuban Ranganath, Connor McManigal, and Peyton Politewicz

*View the project here:* https://connormcmanigal.github.io/Traveling-Salesman-BranchandBoundDFS-StochasticLocalSearch/report.pdf

## The Traveling Salesman Problem:
- The bidirectional traveling salesman problem involves finding the most efficient and optimal route that visits a set of nodes and returns to the starting node, minimizing the total traversal cost.
- Unlike the traditional TSP, this variant considers bidirectional connections between nodes, which introduces additional complexity.

### Implementing Branch and Bound Depth-First Search:
- The Branch and Bound Depth-First Search algorithm used in this project utilizes a systematic approach to combinitorial optimization that systematically explores the search space while using upper and lower bounds to prune the branches that cannot lead to an optimal solution.
- We created a greedy least path heuristic that finds the least-cost path by greedily selecting the nearest unvisited node at each step, aiming to minimize the total cost of traversal in the graph.
- The Depth-First Search strategy efficiently traverses the search space, making it a suitable technique for solving this complex optimization problem.

### Implementing Stochastic Local Search:
- In addition to the BnB DFS method, this project implements a Stochastic Local Search algorithm to solve the TSP.
- SLS algorithms leverage randomness to explore the search space and determines the neighborhood of a solution by identifying states or solutions that are adjacent to the current known solution.
- Our SLS algorithm iteratively explores different permutations of nodes, attempting to minimize the total cost of a tour, while incorporating variance calculation and randomness to escape local optima.
- Our approach of developing a SLS algorithm can strategically circumvent issues that traditional local search algorthms encounter in regards to local minima and maxima.

#### Build Instructions/How to Set Up:
- Clone the repository (git clone <https://github.com/DrinkableBook/TSP--heuritics->)
- Open "group_4_TSP_source.py"
- Install dependencies (numpy, random, statistics, tracemalloc, time)
- Run the code (Note on changing search space: run the problem generator with desired input and/or change the " .out" file to the desired search space)
