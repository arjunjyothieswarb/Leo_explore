import heapq
from math import hypot, sqrt
import jax.numpy as jnp


def heuristic(a, b):
    return sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

# def thicken_wall_jax(grid, radius=3):
#     rows, cols = grid.shape
#     thick_grid = jnp.array(grid)
#     # 使用JAX的数组操作来更新网格
#     for r in range(rows):
#         for c in range(cols):
#             if grid[r, c] == 100:
#                 # 确保索引在边界内
#                 min_r = max(r - radius, 0)
#                 max_r = min(r + radius + 1, rows)
#                 min_c = max(c - radius, 0)
#                 max_c = min(c + radius + 1, cols)
#                 # 更新网格区域
#                 thick_grid = thick_grid.at[min_r:max_r, min_c:max_c].set(100)
#     return thick_grid

def is_near_wall_jax(grid, position, distance):
    rows, cols = grid.shape
    start_row, start_col = position
    min_r = max(start_row - distance, 0)
    max_r = min(start_row + distance + 1, rows)
    min_c = max(start_col - distance, 0)
    max_c = min(start_col + distance + 1, cols)
    # 检查指定区域内是否有墙壁
    return jnp.any(grid[min_r:max_r, min_c:max_c] == 100)

def a_star_search(grid, start, goal, safety_distance=10):

    # grid = thicken_wall_jax(grid)

    if grid[start[0]][start[1]] == 100 or grid[goal[0]][goal[1]] == 100:
        return []  # 如果起点或终点是墙，则立即返回空路径

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
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

# def a_star_search(grid, start, goal):

#     # grid = thicken_walls(grid)

#     if grid[start[0]][start[1]] == 100 or grid[goal[0]][goal[1]] == 100:
#         return []  # 如果起点或终点是墙，则立即返回空路径

#     neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
#     close_set = set()
#     came_from = {}
#     gscore = {start: 0}
#     fscore = {start: heuristic(start, goal)}
#     oheap = []

#     heapq.heappush(oheap, (fscore[start], start))

#     while oheap:
#         current = heapq.heappop(oheap)[1]

#         if current == goal:
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             return path[::-1]

#         close_set.add(current)
#         for i, j in neighbors:
#             neighbor = (current[0] + i, current[1] + j)
#             if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
#                 if grid[neighbor[0]][neighbor[1]] > 0:
#                     continue  # Skip if the cell is a wall
#             else:
#                 continue  # Skip out of bounds

#             tentative_g_score = gscore[current] + heuristic(current, neighbor)
#             if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
#                 continue

#             if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
#                 came_from[neighbor] = current
#                 gscore[neighbor] = tentative_g_score
#                 fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
#                 heapq.heappush(oheap, (fscore[neighbor], neighbor))

#     return False  # Return False if no path is found

# def heuristic(a, b):
#     # return abs(b[0] - a[0]) + abs(b[1] - a[1])
#     return hypot(b[0] - a[0], b[1] - a[1])
#     # return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

# def a_star_search(grid, start, goal):
#     # Directions for moving in the grid (up, down, left, right)
#     # neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
#     delta = 1
#     neighbors = [(0, delta), (delta, 0), (0, -delta), (-delta, 0), (delta, delta), (delta, -delta), (-delta, -delta), (-delta, delta)]
#     # neighbors = [(0, delta), (delta, 0), (0, -delta), (-delta, 0)]
#     close_set = set()
#     came_from = {}
#     gscore = {start: 0}
#     fscore = {start: heuristic(start, goal)}
#     oheap = []

#     heapq.heappush(oheap, (fscore[start], start))
    
#     while oheap:
#         current = heapq.heappop(oheap)[1]

#         if current == goal:
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             return path[::-1]

#         close_set.add(current)
#         for i, j in neighbors:
#             neighbor = current[0] + i, current[1] + j
            
#             if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
#                 # Skip if the cell is an obstacle
#                 if grid[neighbor[0]][neighbor[1]] > 0:
#                     continue
#             else:
#                 # Out of grid bounds
#                 continue

#             tentative_g_score = gscore[current] + 1
#             if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
#                 continue

#             if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
#                 came_from[neighbor] = current
#                 gscore[neighbor] = tentative_g_score
#                 fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
#                 heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
#     return False

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
