import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CourseWorkCode:
    def __init__(self, cid, path):
        self.__set_seed(cid)
        self.x, self.y = self.__read_data(path)
        self.x_label, self.y_label = 'Time', 'Wavelength (nm)'

    def __set_seed(self, cid):
        np.random.seed(cid)

    def __read_data(self, path):
        df = pd.read_csv(path)
        x, y = df['X'].to_numpy(), df['Y'].to_numpy()
        return x, y

    def plot_hist_box(self, bins=10):
        fig, ax = plt.subplots(2, sharex=True, gridspec_kw={
                               'height_ratios': [1, 3]})
        # # Create a boxplot of the y values
        ax[0].boxplot(self.y, vert=False)
        ax[0].set_title('Boxplot')

        # Create a histogram of the x values
        ax[1].hist(self.y, bins=bins, fc='lightblue', ec='k')
        ax[1].set_xlabel(self.y_label)
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Histogram')

        fig.tight_layout()

        # Display the plot
        plt.show()


cid = 1524231
path = 'Statistics/Coursework/rrn18.csv'
cw = CourseWorkCode(cid, path)

# Q1a: Plot a histogram and boxplot of the x and y values
cw.plot_hist_box(bins=10)
