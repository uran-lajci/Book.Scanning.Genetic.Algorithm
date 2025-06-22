import glob
import os
from concurrent.futures import ProcessPoolExecutor

from models import Parser
from models.initial_solution import InitialSolution
from models.genetic_solver import GeneticSolver

INPUT_INSTANCES_DIR = 'input'
OUTPUT_INSTANCES_DIR = 'output'

MINUTES_TO_RUN = 10
MAX_ITERATIONS = 1000
NUM_CORES = 50


def run_solver(version: str, instance_path: str) -> None:
    output_sub_dir = os.path.join(OUTPUT_INSTANCES_DIR, version)
    os.makedirs(output_sub_dir, exist_ok=True)

    parser = Parser(instance_path)
    instance = parser.parse()
    initial_solution = InitialSolution.generate_initial_solution(instance)
    genetic_solver = GeneticSolver(initial_solution=initial_solution, 
                                    instance=instance,
                                    time_limit_sec=MINUTES_TO_RUN * 60)
    solution = genetic_solver.solve()
    score = solution.fitness_score

    instance_name = os.path.basename(instance_path)
    print(instance_name, score, f'version: {version}')
    output_file = os.path.join(output_sub_dir, instance_name)
    solution.export(output_file)


def main():
    instance_paths = glob.glob(f'{INPUT_INSTANCES_DIR}/*.txt')
    jobs = []

    for v in range(1, 6):
        version = f'v{v}'
        for path in instance_paths:
            jobs.append((version, path))

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(run_solver, version, path) for version, path in jobs]

        for future in futures:
            future.result()  


if __name__ == '__main__':
    main()
