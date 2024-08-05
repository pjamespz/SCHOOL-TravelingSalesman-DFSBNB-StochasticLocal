# **Bidirectional Traveling Salesman Problem(TSP)**
- Peyton Politewicz, Shuban Ranganath, Connor McManigal

*View the project here:* https://connormcmanigal.github.io/Traveling-Salesman-BranchandBoundDFS-StochasticLocalSearch/report.pdf

## The Traveling Salesman Problem:
- The bidirectional traveling salesman problem involves finding the most efficient and optimal route that visits a set of nodes and returns to the starting node, minimizing the total traversal cost.

### Implementing Stochastic Local Search:
- I implemented our SLS algorithm for this project, colloquially a 'maximin gambler' approach.
- The least-variant location on the path is determined with some quick preprocessing.
- Then, this point is used as a 'cycle' point for the algorithm to rapidly test other nodes.
- Since we know we stand to lose the least, on average, by cycling through this point, we are free to test other local changes that have far more to gain.
- Out of approximately 40 teams, this approach performed second best in the cohort on provided test data, earning recognition and extra credit.
