import heapq
from math import hypot, sqrt
import jax.numpy as jnp


def heuristic(a, b):
    return sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def is_near_wall_jax(grid, position, distance):
    rows, cols = grid.shape
    start_row, start_col = position
    min_r = max(start_row - distance, 0)
    max_r = min(start_row + distance + 1, rows)
    min_c = max(start_col - distance, 0)
    max_c = min(start_col + distance + 1, cols)

    return jnp.any(grid[min_r:max_r, min_c:max_c] == 100)

def a_star_search(grid, start, goal, safety_distance=0):



    if grid[start[0]][start[1]] == 100 or grid[goal[0]][goal[1]] == 100:
        return [] 

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    # neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and not is_near_wall_jax(grid, neighbor, safety_distance):
                if grid[neighbor[0]][neighbor[1]] > 0:
                    continue  # Skip if the cell is a wall

                tentative_g_score = gscore[current] + heuristic(current, neighbor)
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

# Example usage
# grid = [
#     [1, 1, 1, 100, 1],
#     [1, 100, 1, 100, 1],
#     [-1, -1, 1, 1, 1],
#     [-1, 1, 1, 100, 1],
#     [-1, -1, 1, 1, 1]
# ]
# start = (0, 0)
# goal = (4, 4)
# path = a_star_search(grid, start, goal)
# print(path)
