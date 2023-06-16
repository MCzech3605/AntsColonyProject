import matplotlib.pyplot as plt  # type: ignore


def plot_comparison_results(time_tab: list[float],
                            ans_tab: list[float],
                            number_of_algorithms: int,
                            graph_size: int) -> None:

    fig, ax = plt.subplots()
    ax.grid()
    if number_of_algorithms == 3:
        ax.bar([1, 2, 3], time_tab, tick_label=["Alg2", "Alg3", "Classic"])
    else:
        ax.bar([1, 2, 3, 4], time_tab, tick_label=["Alg1", "Alg2", "Alg3", "Classic"])
    ax.set_title("Average time taken")
    ax.set_ylabel("Time [ms]")
    fig.show()
    fig.savefig(f"../figures/Time_lin_{number_of_algorithms}_{graph_size}.png")
    ax.set_yscale('log')
    fig.show()
    fig.savefig(f"../figures/Time_log_{number_of_algorithms}_{graph_size}.png")

    fig, ax = plt.subplots()
    ax.grid()
    if number_of_algorithms == 3:
        ax.bar([1, 2, 3], ans_tab, tick_label=["Alg2", "Alg3", "Classic"])
    else:
        ax.bar([1, 2, 3, 4], ans_tab, tick_label=["Alg1", "Alg2", "Alg3", "Classic"])
    ax.set_title("Average flow reached")
    ax.set_ylabel("Flow [units]")
    fig.show()
    fig.savefig(f"../figures/Flow_lin_{number_of_algorithms}_{graph_size}.png")


def plot_accuracy_by_iterations(results_ant: list[float],
                                optimal_result: int,
                                iterations: int) -> None:

    plt.plot(results_ant, label="Ant")
    plt.plot([optimal_result for _ in range(iterations)], label="Ford-Fulkerson")
    plt.xlabel("Iterations")
    plt.ylabel("Maximum flow found")
    plt.title("Accuracy of an ant algorithm by iterations")
    plt.legend()
    plt.savefig("../figures/convergence.png")
    plt.show()
