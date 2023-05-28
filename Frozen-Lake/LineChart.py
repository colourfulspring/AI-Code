import matplotlib.pyplot as plt


class LineChart(object):
    def __init__(self):
        pass

    def add_line(self, data, label):
        x = range(len(data))
        plt.plot(x, data, label=label)

    def draw_line_chart(self, xlabel, ylabel, title):
        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Add legend
        plt.legend()

        # Display the chart
        plt.show()


