from src.plotting import plot_accuracy_by_iterations, plot_comparison_results
from src.trials import run_trials, run_accuracy_trials


def main() -> None:
    results_ant, optimal_result = run_accuracy_trials(20, 200, 50, 0.2, (3, 25))
    plot_accuracy_by_iterations(results_ant, optimal_result, 200)

    time_tab, ans_tab = run_trials(20, 5, 200, 10, 0.3, (2, 10))
    plot_comparison_results(time_tab, ans_tab, 3, 10)

    time_tab, ans_tab = run_trials(20, 5, 150, 50, 0.2, (3, 25))
    plot_comparison_results(time_tab, ans_tab, 3, 50)


if __name__ == '__main__':
    main()
