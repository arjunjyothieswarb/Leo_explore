import heapq
from math import hypot

def heuristic(a, b):
    # Using Manhattan Distance as the heuristic
    # return abs(b[0] - a[0]) + abs(b[1] - a[1])
    return hypot(b[0] - a[0], b[1] - a[1])

def a_star_search(grid, start, goal):
    # Directions for moving in the grid (up, down, left, right)
    # neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    delta = 1
    neighbors = [(0, delta), (delta, 0), (0, -delta), (-delta, 0), (delta, delta), (delta, -delta), (-delta, -delta), (-delta, delta)]
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
            neighbor = current[0] + i, current[1] + j
            
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                # Skip if the cell is an obstacle or unknown
                if grid[neighbor[0]][neighbor[1]] == 100:
                    continue
            else:
                # Out of grid bounds
                continue

            tentative_g_score = gscore[current] + 1
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
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
