import math
import random
import copy
from typing import List, Tuple
import matplotlib.pyplot as plt


#################################################################

import math
from typing import List, Tuple

class Package:
    def __init__(self, package_id: int, destination: Tuple[float, float], weight: float, priority: int):
        self.id = package_id
        self.destination = destination  # (x, y)
        self.weight = weight
        self.priority = priority

    def __repr__(self):
        return f"Package(id={self.id}, dest={self.destination}, weight={self.weight}, priority={self.priority})"

class Vehicle:
    def __init__(self, vehicle_id: int, capacity: float):
        self.id = vehicle_id
        self.capacity = capacity
        self.packages: List[Package] = []
        self.route: List[Tuple[float, float]] = [(0, 0)]  # Start at the shop (0,0)

    def add_package(self, package: Package) -> bool:
        if self.current_load() + package.weight <= self.capacity:
            self.packages.append(package)
            self.route.append(package.destination)
            return True
        return False

    def current_load(self) -> float:
        return sum(pkg.weight for pkg in self.packages)

    def total_distance(self) -> float:
        dist = 0.0
        for i in range(1, len(self.route)):
            dist += euclidean_distance(self.route[i - 1], self.route[i])
        dist += euclidean_distance(self.route[-1], (0, 0))  # return to shop
        return dist

    def __repr__(self):
        return f"Vehicle(id={self.id}, load={self.current_load()}/{self.capacity}, packages={len(self.packages)})"

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
############################################################################

def create_packages_from_user() -> List[Package]:
    packages = []
    num_packages = int(input("Enter number of packages: "))

    for i in range(num_packages):
        print(f"\nPackage #{i + 1}")
        x = float(input("  Destination X (0–100): "))
        y = float(input("  Destination Y (0–100): "))
        weight = float(input("  Weight (kg): "))
        priority = int(input("  Priority (1=high, 5=low): "))

        pkg = Package(i, (x, y), weight, priority)
        packages.append(pkg)

    return packages

def create_vehicles_from_user() -> List[Vehicle]:
    vehicles = []
    num_vehicles = int(input("\nEnter number of delivery vehicles: "))

    for i in range(num_vehicles):
        capacity = float(input(f"  Capacity for Vehicle #{i + 1} (kg): "))
        vehicle = Vehicle(i, capacity)
        vehicles.append(vehicle)

    return vehicles

######################################################################

import copy
import random

def initial_solution(packages: List[Package], vehicles: List[Vehicle]) -> List[Vehicle]:
    solution = copy.deepcopy(vehicles)
    # Sort packages by priority and proximity to depot
    sorted_packages = sorted(
        packages,
        key=lambda p: (p.priority, euclidean_distance((0, 0), p.destination))
    )
    for pkg in sorted_packages:
        assigned = False
        for vehicle in solution:
            if vehicle.add_package(pkg):
                assigned = True
                break
        if not assigned:
            print(f"Warning: Package {pkg.id} (Priority {pkg.priority}) could not be assigned.")
    return solution

def calculate_total_distance(vehicles: List[Vehicle]) -> float:
    return sum(vehicle.total_distance() for vehicle in vehicles)

def generate_neighbor(solution: List[Vehicle]) -> List[Vehicle]:
    """
    Generate a neighboring solution by randomly swapping or moving a package between vehicles.
    """
    neighbor = copy.deepcopy(solution)

    if len(neighbor) < 2:
        return neighbor

    v1, v2 = random.sample(neighbor, 2)

    if not v1.packages and not v2.packages:
        return neighbor

    operation = random.choice(["swap", "move"])

    if operation == "swap" and v1.packages and v2.packages:
        p1 = random.choice(v1.packages)
        p2 = random.choice(v2.packages)

        v1_new_load = v1.current_load() - p1.weight + p2.weight
        v2_new_load = v2.current_load() - p2.weight + p1.weight

        if v1_new_load <= v1.capacity and v2_new_load <= v2.capacity:
            v1.packages.remove(p1)
            v2.packages.remove(p2)
            v1.packages.append(p2)
            v2.packages.append(p1)

    elif operation == "move":
        source, target = (v1, v2) if v1.packages else (v2, v1)
        if source.packages:
            pkg = random.choice(source.packages)
            if target.current_load() + pkg.weight <= target.capacity:
                source.packages.remove(pkg)
                target.packages.append(pkg)

    for v in neighbor:
        v.route = [(0, 0)] + [p.destination for p in v.packages]

    return neighbor

############################################################################3

def simulated_annealing(
    packages: List[Package],
    vehicles: List[Vehicle],
    initial_temp: float = 1000,
    cooling_rate: float = 0.95,
    stopping_temp: float = 1.0,
    iterations_per_temp: int = 100
) -> Tuple[List[Vehicle], float]:
    current_solution = initial_solution(packages, vehicles)
    current_cost = calculate_total_distance(current_solution)
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost

    temperature = initial_temp

    while temperature > stopping_temp:
        for _ in range(iterations_per_temp):
            neighbor = generate_neighbor(current_solution)
            neighbor_cost = calculate_total_distance(neighbor)
            delta = neighbor_cost - current_cost

            if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temperature):
                current_solution = neighbor
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost

        temperature *= cooling_rate  # Reduce temperature

    return best_solution, best_cost

#######################################################################

def get_sa_parameters_from_user() -> Tuple[float, float, float, int]:
    print("\n--- Simulated Annealing Parameters Setup ---")

    try:
        initial_temp = float(input("Initial Temperature [default=1000]: ") or 1000)
        cooling_rate = float(input("Cooling Rate (0.90–0.99) [default=0.95]: ") or 0.95)
        stopping_temp = float(input("Stopping Temperature [default=1.0]: ") or 1.0)
        iterations_per_temp = int(input("Iterations per Temperature [default=100]: ") or 100)
    except ValueError:
        print("Invalid input. Using default parameters.")
        return 1000, 0.95, 1.0, 100

    return initial_temp, cooling_rate, stopping_temp, iterations_per_temp

###############################################################################
def plot_solution(vehicles: List[Vehicle]):
    colors = plt.colormaps.get_cmap("tab10")

    plt.figure(figsize=(10, 8))
    plt.title("Vehicle Routes from Shop (0,0)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    for i, vehicle in enumerate(vehicles):
        if not vehicle.packages:
            continue

        x_coords = [0] + [pkg.destination[0] for pkg in vehicle.packages] + [0]
        y_coords = [0] + [pkg.destination[1] for pkg in vehicle.packages] + [0]

        color = colors(i % 10)
        plt.plot(x_coords, y_coords, color=color, marker='o', label=f"Vehicle {vehicle.id} ({vehicle.total_distance():.1f} km)")

        for pkg in vehicle.packages:
            label = f"ID:{pkg.id}\nP:{pkg.priority}"
            plt.text(pkg.destination[0], pkg.destination[1], label, fontsize=8, ha='left', va='bottom')

    plt.scatter(0, 0, color='black', s=100, label="Shop (0,0)", marker='s')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


############################################################################

# --------------------------------------------
# 🔹 8. Genetic Algorithm Setup
# --------------------------------------------

# Step 1: Generate Initial Population
# 👉 def generate_initial_population(packages, vehicles, population_size):
#     - Generate a list of candidate solutions (vehicle assignments and routes)

import copy
import random

def generate_initial_population(packages, vehicles, population_size):
    population = []

    for _ in range(population_size):
        individual = [copy.deepcopy(v) for v in vehicles]
        unassigned_packages = packages[:]
        random.shuffle(unassigned_packages)

        for pkg in unassigned_packages:
            assigned = False
            vehicle_order = random.sample(individual, len(individual))
            for vehicle in vehicle_order:
                if vehicle.current_load() + pkg.weight <= vehicle.capacity:
                    vehicle.add_package(pkg)
                    assigned = True
                    break
            if not assigned:
                break

        total_assigned = sum(len(v.packages) for v in individual)
        if total_assigned == len(packages):
            population.append(individual)

    return population


# Step 2: Fitness Function
# 👉 def fitness(solution):
#     - Calculate total distance (lower is better)
import math

def fitness(individual):
    total_distance = sum(vehicle.total_distance() for vehicle in individual)
    return -total_distance  # Negative for minimization


# Step 3: Selection Function
# 👉 def selection(population, fitness_scores):
#     - Select two parent solutions based on fitness (e.g., roulette or tournament)
def selection(population, fitness_scores, tournament_size=3):
    selected_parents = []

    for _ in range(2):  # Select 2 parents
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])  # Highest fitness wins
        selected_parents.append(winner[0])  # Add only the individual, not its score

    return selected_parents[0], selected_parents[1]


# Step 4: Crossover Function
# 👉 def crossover(parent1, parent2):
#     - Combine parts of both parents into two children
def crossover(parent1, parent2):
    # Collect all unique packages from both parents
    all_packages = []
    for v in parent1:
        all_packages.extend(v.packages)
    for v in parent2:
        all_packages.extend(v.packages)
    # Deduplicate (each package should appear once)
    unique_packages = list({pkg.id: pkg for pkg in all_packages}.values())
    random.shuffle(unique_packages)

    def create_child():
        # Create new vehicles with same IDs and capacities
        child = [Vehicle(v.id, v.capacity) for v in parent1]
        # Assign packages using Vehicle.add_package()
        for pkg in unique_packages:
            assigned = False
            for vehicle in random.sample(child, len(child)):  # Try vehicles in random order
                if vehicle.add_package(pkg):
                    assigned = True
                    break
            if not assigned:
                return None  # Child is invalid
        return child

    child1 = create_child()
    child2 = create_child()
    return child1, child2

# Step 5: Mutation Function
# 👉 def mutation(solution, mutation_rate):
#     - Randomly swap package assignments or shuffle routes
def mutate(individual, mutation_rate=0.1):
    for vehicle in individual:
        if random.random() < mutation_rate and vehicle.packages:
            # Select a package to move
            pkg = random.choice(vehicle.packages)
            # Choose a different vehicle
            other_vehicles = [v for v in individual if v != vehicle]
            if not other_vehicles:
                continue
            other_vehicle = random.choice(other_vehicles)
            # Check if the other vehicle can accommodate the package
            if other_vehicle.current_load() + pkg.weight <= other_vehicle.capacity:
                # Remove from current vehicle and add to other
                vehicle.packages.remove(pkg)
                other_vehicle.packages.append(pkg)
                # Update routes for both vehicles
                vehicle.route = [(0, 0)] + [p.destination for p in vehicle.packages]
                other_vehicle.route = [(0, 0)] + [p.destination for p in other_vehicle.packages]

# Step 6: GA Main Loop
# 👉 def genetic_algorithm(packages, vehicles, population_size, mutation_rate, generations):
#     - Full loop: generate → evaluate → select → crossover → mutate
def genetic_algorithm(packages, vehicles, population_size=50, generations=100, mutation_rate=0.1):
    population = generate_initial_population(packages, vehicles, population_size)
    best_solution = None
    best_fitness_score = float("-inf")

    for gen in range(generations):
        fitness_scores = [fitness(ind) for ind in population]

        # Track best solution
        for i, score in enumerate(fitness_scores):
            if score > best_fitness_score:
                best_fitness_score = score
                best_solution = population[i]

        new_population = []

        while len(new_population) < population_size:
            # Selection
            parent1, parent2 = selection(population, fitness_scores)

            # Crossover
            child1, child2 = crossover(parent1, parent2)
            if child1 and child2:
                # Mutation
                mutate(child1, mutation_rate)
                mutate(child2, mutation_rate)

                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)

        population = new_population

    total_distance = -fitness(best_solution)  # Convert back from negative
    return best_solution, total_distance

#################################################################################

def get_ga_parameters_from_user():
    print("\nEnter Genetic Algorithm parameters:")

    while True:
        try:
            population_size = int(input("Population Size (e.g., 50): "))
            generations = int(input("Number of Generations (e.g., 100): "))
            mutation_rate = float(input("Mutation Rate (0.01 to 0.1): "))
            if population_size > 0 and generations > 0 and 0.0 <= mutation_rate <= 1.0:
                return population_size, generations, mutation_rate
            else:
                print("Please enter valid values.")
        except ValueError:
            print("Invalid input. Please enter numeric values.")

####################################################################################

def assign_packages_to_vehicles(package_order, vehicles):
    assigned_vehicles = [Vehicle(v.capacity) for v in vehicles]

    current_vehicle_index = 0
    for pkg in package_order:
        assigned = False
        while not assigned and current_vehicle_index < len(assigned_vehicles):
            vehicle = assigned_vehicles[current_vehicle_index]
            if vehicle.can_add(pkg):
                vehicle.add_package(pkg)
                assigned = True
            else:
                current_vehicle_index += 1

        if not assigned:
            # If we can't assign due to capacity limits, start over (not optimal, but prevents crash)
            current_vehicle_index = 0

    return assigned_vehicles

###############################################################################################

# Step 7: Plot Result
# 👉 plot_solution(best_solution)
import matplotlib.pyplot as plt

def plot_ga_solution(solution, generation=None, total_distance=None):
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap("tab20")  # Get the colormap object

    for idx, vehicle in enumerate(solution):
        if not vehicle.packages:
            continue

        # Extract coordinates from vehicle's route (which includes the depot)
        x = [point[0] for point in vehicle.route]
        y = [point[1] for point in vehicle.route]

        # Get color from colormap
        color = cmap(idx % cmap.N)  # Cycle through colors if needed

        # Plot route
        plt.plot(x, y, marker='o', linestyle='-', color=color, label=f"Vehicle {vehicle.id}")

        # Annotate packages
        for pkg in vehicle.packages:
            label = f"ID:{pkg.id} P:{pkg.priority}"
            plt.text(pkg.destination[0], pkg.destination[1], label, fontsize=8, ha='left', va='bottom')

    plt.scatter(0, 0, color='black', s=100, label="Shop (0,0)", marker='s')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    title = "Genetic Algorithm Vehicle Routes"
    if generation is not None:
        title += f" (Gen {generation})"
    plt.title(title)
    plt.legend()
    if total_distance is not None:
        plt.suptitle(f"Total Distance: {total_distance:.2f} km", fontsize=10, y=0.92)
    plt.grid(True)
    plt.show()

#################################################################

# --------------------------------------------
# 🔹 8. Main Program Execution
# --------------------------------------------
def run_simulated_annealing():
    print("\n🚚 AI-Based Package Delivery Optimizer (Simulated Annealing)\n")

    # Step 1: Get user-defined packages and vehicles
    packages = create_packages_from_user()
    vehicles = create_vehicles_from_user()

    # Step 2: Get Simulated Annealing parameters
    initial_temp, cooling_rate, stopping_temp, iterations_per_temp = get_sa_parameters_from_user()

    # Step 3: Run Simulated Annealing
    best_solution, best_cost = simulated_annealing(
        packages,
        vehicles,
        initial_temp,
        cooling_rate,
        stopping_temp,
        iterations_per_temp
    )

    # Step 4: Output results
    print(f"\n✅ Simulated Annealing complete. Total Distance: {best_cost:.2f} km")
    for vehicle in best_solution:
        print(vehicle)

    # Step 5: Visualize result
    plot_solution(best_solution)


def run_genetic_algorithm():
    print("\n🧬 AI-Based Package Delivery Optimizer (Genetic Algorithm)\n")

    # Step 1: Get user-defined packages and vehicles
    packages = create_packages_from_user()
    vehicles = create_vehicles_from_user()

    # Step 2: Get Genetic Algorithm parameters
    population_size, generations, mutation_rate = get_ga_parameters_from_user()

    # Step 3: Run Genetic Algorithm
    best_solution, best_cost = genetic_algorithm(
        packages,
        vehicles,
        population_size,
        generations,
        mutation_rate
    )

    # Step 4: Output results
    print(f"\n✅ Genetic Algorithm complete. Total Distance: {best_cost:.2f} km")
    for vehicle in best_solution:
        print(vehicle)

    # Step 5: Visualize result uniquely
    plot_ga_solution(best_solution, generation=generations, total_distance=best_cost)


def main():
    print("🚀 AI-Based Package Delivery Optimization")
    print("Choose Optimization Algorithm:")
    print("1. Simulated Annealing")
    print("2. Genetic Algorithm")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        run_simulated_annealing()
    elif choice == "2":
        run_genetic_algorithm()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()