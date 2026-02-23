import matplotlib.pyplot as plt


class PlotStressVsH:

    def __init__(
        self,
        shapes,
        h_values_list,
        stress_values_list,
        labels=None
    ):
        if not (
            len(shapes) ==
            len(h_values_list) ==
            len(stress_values_list)
        ):
            raise ValueError("All input lists must have same length.")

        self.shapes = shapes
        self.h_values_list = h_values_list
        self.stress_values_list = stress_values_list

        if labels is None:
            self.labels = [
                type(shape).__name__
                for shape in shapes
            ]
        else:
            self.labels = labels

    def plot(self, figsize=(7, 5), marker="o"):

        fig, ax = plt.subplots(figsize=figsize)

        cmap = plt.cm.get_cmap("tab10")

        for i, (h_vals, stress_vals, label) in enumerate(
            zip(self.h_values_list,
                self.stress_values_list,
                self.labels)
        ):
            color = cmap(i)

            ax.plot(
                h_vals,
                stress_vals,
                marker=marker,
                color=color,
                label=label
            )

        ax.set_xlabel("h")
        ax.set_ylabel("S-Stress")
        ax.set_title("Stress vs h")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        return fig, ax