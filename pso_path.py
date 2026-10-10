"""
Swarm Intelligence Lab - Assignment 1
PSO-based path planning on a 2D grid with obstacles.
Seed (roll number): 61
"""
import random
import math
from collections import deque

import matplotlib.pyplot as plt

# ---------------- Settings ----------------
SEED = 61              # roll number 061
GRID_SIZE = 20         # 20 x 20 grid
OBSTACLE_RATIO = 0.20  # 20% of cells are blocked
MIN_START_GOAL_DIST = 14

NUM_WAYPOINTS = 8      # points between start and goal
NUM_PARTICLES = 80
ITERATIONS = 500
W, C1, C2 = 0.7, 1.5, 1.5   # inertia, cognitive, social
PENALTY = 40           # cost added for each blocked sample point
STEP = 0.25            # sampling distance along a segment for collision check
# ------------------------------------------


# ---------- Grid / obstacle generation ----------
def reachable(obstacles, start, goal):
    """BFS check that goal can be reached from start (4-neighbour)."""
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if (0 <= n[0] < GRID_SIZE and 0 <= n[1] < GRID_SIZE
                    and n not in obstacles and n not in seen):
                seen.add(n)
                queue.append(n)
    return False


def generate_problem():
    """Create obstacles, start and goal from the seed (no hardcoding)."""
    random.seed(SEED)
    cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
    while True:
        obstacles = set(random.sample(cells, int(OBSTACLE_RATIO * len(cells))))
        free = [c for c in cells if c not in obstacles]
        start, goal = random.sample(free, 2)
        far_enough = math.dist(start, goal) >= MIN_START_GOAL_DIST
        if far_enough and reachable(obstacles, start, goal):
            return obstacles, start, goal


# ---------- Path evaluation ----------
def to_points(particle, start, goal):
    """Particle = flat list [x1,y1,x2,y2,...]. Returns full path with cell centres."""
    s = (start[0] + 0.5, start[1] + 0.5)
    g = (goal[0] + 0.5, goal[1] + 0.5)
    mid = [(particle[i], particle[i + 1]) for i in range(0, len(particle), 2)]
    return [s] + mid + [g]


def count_collisions(path, obstacles):
    """Walk along each segment and count sample points inside blocked cells."""
    hits = 0
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        d = math.dist((x1, y1), (x2, y2))
        n = max(1, int(d / STEP))
        for i in range(n + 1):
            t = i / n
            cell = (int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)))
            if cell in obstacles:
                hits += 1
    return hits


def path_length(path):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def fitness(particle, obstacles, start, goal):
    path = to_points(particle, start, goal)
    return path_length(path) + PENALTY * count_collisions(path, obstacles)


# ---------- PSO ----------
def pso(obstacles, start, goal):
    dim = NUM_WAYPOINTS * 2
    lo, hi = 0.0, GRID_SIZE - 0.001

    def clamp(v):
        return max(lo, min(hi, v))

    # initialise particles around the straight line start -> goal
    particles, velocities = [], []
    for _ in range(NUM_PARTICLES):
        p = []
        for k in range(1, NUM_WAYPOINTS + 1):
            t = k / (NUM_WAYPOINTS + 1)
            x = start[0] + 0.5 + t * (goal[0] - start[0]) + random.uniform(-5, 5)
            y = start[1] + 0.5 + t * (goal[1] - start[1]) + random.uniform(-5, 5)
            p += [clamp(x), clamp(y)]
        particles.append(p)
        velocities.append([random.uniform(-1, 1) for _ in range(dim)])

    pbest = [p[:] for p in particles]
    pbest_cost = [fitness(p, obstacles, start, goal) for p in particles]
    g_idx = pbest_cost.index(min(pbest_cost))
    gbest, gbest_cost = pbest[g_idx][:], pbest_cost[g_idx]
    history = [gbest_cost]

    for it in range(ITERATIONS):
        for i in range(NUM_PARTICLES):
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (W * velocities[i][d]
                                    + C1 * r1 * (pbest[i][d] - particles[i][d])
                                    + C2 * r2 * (gbest[d] - particles[i][d]))
                particles[i][d] = clamp(particles[i][d] + velocities[i][d])

            cost = fitness(particles[i], obstacles, start, goal)
            if cost < pbest_cost[i]:
                pbest[i], pbest_cost[i] = particles[i][:], cost
                if cost < gbest_cost:
                    gbest, gbest_cost = particles[i][:], cost
        history.append(gbest_cost)
        if (it + 1) % 50 == 0:
            print(f"Iteration {it + 1:4d} | best cost = {gbest_cost:.3f}")

    return to_points(gbest, start, goal), gbest_cost, history


# ---------- Visualisation ----------
def plot(obstacles, start, goal, path, history, collisions):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    for (x, y) in obstacles:
        ax1.add_patch(plt.Rectangle((x, y), 1, 1, color="black"))
    ax1.scatter(start[0] + 0.5, start[1] + 0.5, c="green", s=150, label="Start", zorder=3)
    ax1.scatter(goal[0] + 0.5, goal[1] + 0.5, c="red", s=150, marker="*", label="Goal", zorder=3)
    xs, ys = zip(*path)
    ax1.plot(xs, ys, "b-o", markersize=4, label="PSO path")
    ax1.set_xlim(0, GRID_SIZE)
    ax1.set_ylim(0, GRID_SIZE)
    ax1.set_xticks(range(GRID_SIZE + 1))
    ax1.set_yticks(range(GRID_SIZE + 1))
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")
    ax1.set_title(f"PSO path (seed={SEED}, length={path_length(path):.2f}, collisions={collisions})")
    ax1.legend(loc="lower left")

    ax2.plot(history)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best cost")
    ax2.set_title("Convergence")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("result.png", dpi=150)
    plt.show()


def main():
    obstacles, start, goal = generate_problem()
    print(f"Seed: {SEED} | Grid: {GRID_SIZE}x{GRID_SIZE} | Obstacles: {len(obstacles)}")
    print(f"Start: {start} | Goal: {goal}")

    random.seed(SEED)  # reproducible PSO run
    path, cost, history = pso(obstacles, start, goal)

    collisions = count_collisions(path, obstacles)
    print("\nBest path found:")
    for p in path:
        print(f"  ({p[0]:.2f}, {p[1]:.2f})")
    print(f"Path length: {path_length(path):.3f}")
    print(f"Collisions : {collisions}")
    print(f"Final cost : {cost:.3f}")

    plot(obstacles, start, goal, path, history, collisions)


if __name__ == "__main__":
    main()
