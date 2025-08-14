import numpy as np

def hungarian_algorithm(cost_matrix):
    """
    Hungarian algorithm for solving the assignment problem in O(n^3) time.
    Returns:
        match_rows: list where match_rows[i] = column assigned to row i
        optimal_cost: sum of the costs for the optimal assignment
    """
    cost_matrix = np.array(cost_matrix)
    n = cost_matrix.shape[0]

    # Potentials for rows and columns
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)

    # p[j] = row assigned to column j
    p = np.zeros(n + 1, dtype=int)

    # way[j] = to keep track of the previous column vertex visited.
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
       
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)  #used[j] = True if column j is in the tree/path. In other words, it keeps track of T.

        while True:
      
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
               #Ignore the columns that are already in the tree.

                if not used[j]:
                    cur = cost_matrix[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0     #update the way.
                        
                        
                    if minv[j] < delta:
                        delta = minv[j]
                        
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta    #l(T) = l(T) + delta
                    v[j] -= delta       #l(S)= l(S) - delta
                else:
                    minv[j] -= delta    #update potentials

            j0 = j1 #j0 is the new column that is added to the tree. (i.e. Our current column)
          

            if p[j0] == 0: #If the current column is not assigned to any row, then we have found an augmenting path.
                break

        # Augmenting path construction. Backtrack relabeling of edges in matching.
        while True:
            
            j1 = way[j0]    #Now reuse/relabel j1 such that j1 tells you the previous column vertex visited.
            

            p[j0] = p[j1] #reassign rows according to augmenting path.
         

            j0 = j1
            if j0 == 0:
            
                break
        
       

    # Build result: match_rows[i] = assigned column index
    match_rows = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        match_rows[p[j] - 1] = j - 1

    optimal_cost = cost_matrix[range(n), match_rows].sum()
    return match_rows.tolist(), optimal_cost


# Example usage
import time
# ... existing code ...

# Example usage
if __name__ == "__main__":
    cost_matrix = np.random.randint(1, 10, size=(100, 100))
    
    # Time the algorithm with high precision
    start_time = time.perf_counter()
    assignment, cost = hungarian_algorithm(cost_matrix)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    print("Assignment (row -> col):", assignment)
    print("Optimal cost:", cost)
    print(f"Execution time: {execution_time:.9f} seconds")