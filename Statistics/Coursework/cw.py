import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class CourseWork:
    def __init__(self, cid, path):
        self.__set_seed(cid)
        self.x, self.y = self.__read_data(path)
        self.x_st = self.__standardise(self.x)
        self.data = np.array([self.x, self.y]).T
        self.x_flt, self.y_flt = self.__get_filtered_xy(
            bounds=(10, 850), step=10)
        self.x_flt_st = self.__standardise(self.x_flt)
        self.x_label, self.y_label = 'Time', 'Wavelength (nm)'

    def to_ordinal(self, n: int):
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        else:
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
        return str(n) + suffix

    def __set_seed(self, cid):
        np.random.seed(cid)

    def __read_data(self, path):
        df = pd.read_csv(path)
        x, y = df['X'].to_numpy(), df['Y'].to_numpy()
        return x, y

    def __standardise(self, x):
        return (x - np.mean(x)) / np.std(x)

    def __get_filtered_xy(self, bounds: tuple, step: float):
        x = np.arange(bounds[0], bounds[1]+step, step)
        mask = np.isin(self.x, x)
        y = self.y[mask]
        return x, y

    def __get_data(self, standardise: bool, filter: bool):
        if filter:
            x = self.x_flt_st if standardise else self.x_flt
            y = self.y_flt
        else:
            x = self.x_st if standardise else self.x
            y = self.y
        return x, y

    def __config_plot(self,
                      ax,
                      x_label=None,
                      y_label=None,
                      title=None,
                      legend=False):
        ax.set_title(title) if title else None
        ax.set_xlabel(x_label) if x_label else None
        ax.set_ylabel(y_label) if y_label else None
        ax.legend() if legend else None
        return ax

    def __create_scatter_plot(self,
                              standardise: bool,
                              filter: bool):
        x, y = self.__get_data(standardise, filter)
        filtered_label = 'Filtered ' if filter else ''
        standarised_label = f'Standardised ' if standardise else ''
        fig, ax = plt.subplots()
        ax.scatter(x,
                   y,
                   s=1,
                   label=filtered_label + standarised_label + 'Data')
        return ax

    def plot_hist_box(self, bins=10):
        fig, ax = plt.subplots(2,
                               sharex=True,
                               gridspec_kw={'height_ratios': [1, 3]})
        # # Create a boxplot of the y values
        ax[0].boxplot(self.y, vert=False)
        self.__config_plot(ax[0], title='(a)')
        # Create a histogram of the x values
        ax[1].hist(self.y, bins=bins, fc='lightblue', ec='k')
        self.__config_plot(ax[1], self.y_label, 'Frequency', '(b)')
        fig.tight_layout()
        # Display the plot
        plt.show()

    def summarise_location(self):
        return {
            'mean': np.mean(self.y),
            'trimmed_mean': stats.trim_mean(self.y, 0.1),
            'median': np.median(self.y),
            'std': np.std(self.y),
            'iqr': stats.iqr(self.y)
        }

    def plot_scatter(self, standardise: bool, filter: bool):
        ax = self.__create_scatter_plot(standardise, filter)
        x_label = f'Standardised {self.x_label}' \
            if standardise else self.x_label
        self.__config_plot(ax, x_label, self.y_label)
        plt.show()

    def __get_x_poly(self, x, degree):
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        return poly.fit_transform(x.reshape(-1, 1))

    def fit_n_degree_regression(self,
                                degree: int,
                                standardise: bool,
                                filter: bool):
        x, y = self.__get_data(standardise, filter)
        x_poly = self.__get_x_poly(x, degree)
        model = LinearRegression()
        model.fit(x_poly, y)
        beta = model.coef_
        beta[0] = model.intercept_
        return beta, model.score(x_poly, y)

    def plot_n_degree_regression(self,
                                 degree: int,
                                 standardise: bool,
                                 filter: bool):
        x_label = f'Standardised {self.x_label}' \
            if standardise else self.x_label
        x, y = self.__get_data(standardise, filter)
        beta, score = self.fit_n_degree_regression(
            degree, standardise, filter)
        x_fit = np.linspace(x.min(), x.max(), 100)
        x_fit_poly = self.__get_x_poly(x_fit, degree)
        y_fit = np.dot(x_fit_poly, beta)
        ax = self.__create_scatter_plot(standardise, filter)
        ax.plot(x_fit, y_fit, 'r',
                label=f'{self.to_ordinal(degree)} Degree Fit')
        self.__config_plot(ax, x_label, self.y_label, legend=True)
        plt.show()

    def plot_all_n_degree_regressions(self,
                                      degree_range: tuple,
                                      standardise: bool,
                                      filter: bool):
        x_label = f'Standardised {self.x_label}' \
            if standardise else self.x_label
        x, y = self.__get_data(standardise, filter)
        ax = self.__create_scatter_plot(standardise, filter)
        for degree in range(degree_range[0], degree_range[1]+1):
            beta, score = self.fit_n_degree_regression(
                degree, standardise, filter)
            x_fit = np.linspace(x.min(), x.max(), 100)
            x_fit_poly = self.__get_x_poly(x_fit, degree)
            y_fit = np.dot(x_fit_poly, beta)
            ax.plot(x_fit, y_fit,
                    label=f'{self.to_ordinal(degree)} Degree Fit')
        self.__config_plot(ax, x_label, self.y_label, legend=True)
        plt.show()

    def get_residuals(self, degree: int, standardise: bool, filter: bool):
        x, y = self.__get_data(standardise, filter)
        x_poly = self.__get_x_poly(x, degree)
        beta, score = self.fit_n_degree_regression(
            degree, standardise, filter)
        error = y - np.dot(x_poly, beta)
        return error

    def plot_residuals(self, degree: int, standardise: bool, filter: bool):
        x_label = f'Standardised {self.x_label}' \
            if standardise else self.x_label
        x, y = self.__get_data(standardise, filter)
        error = self.get_residuals(degree, standardise, filter)
        mean_error = np.mean(error)
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[0].scatter(x, error, s=1, label='Residuals')
        ax[0].axhline(mean_error, c='r', ls='--', label='Mean Error')
        self.__config_plot(ax[0],
                           x_label=x_label,
                           y_label='Error (nm)',
                           title='(a)',
                           legend=True)
        ax[1].hist(error, bins=10, fc='lightblue',
                   ec='k', orientation='horizontal')
        ax[1].axhline(mean_error, c='r', ls='--', label='Mean Error')
        self.__config_plot(ax[1],
                           x_label='Frequency',
                           title='(b)',
                           legend=True)
        plt.show()

    def __log_likehood(self, degree: int, standardise: bool, filter: bool):
        # x, y = self.__get_data(standardise, filter)
        error = self.get_residuals(degree, standardise, filter)
        n = len(error)
        squared_error = np.matmul(error.T, error)
        variance = np.std(error)**2
        # variance = np.std(y)**2
        exponent = (-1 / (2 * variance)) * squared_error
        log_likehood = -0.5 * n * np.log(2 * np.pi * variance) + exponent
        return log_likehood

    def get_aic(self, degree: int, standardise: bool, filter: bool):
        log_likehood = self.__log_likehood(degree, standardise, filter)
        q = degree + 2
        aic = 2 * q - 2 * log_likehood
        return aic


def q1_script(cw):
    # Q1a: Plot a histogram and boxplot of the x and y values
    cw.plot_hist_box(bins=10)
    # Q1b: Summarise the location of the data
    print('Location summary of raw data:')
    print(cw.summarise_location())
    # Q1c: Plot a scatter plot of the data
    cw.plot_scatter(standardise=False, filter=False)


def q2_script(cw):
    def __sumarise_and_plot_regression_fit(degree: int,
                                           is_standarised: bool,
                                           is_filtered: bool):
        standarised_label = 'Standarised ' if is_standarised else 'Raw'
        filtered_label = 'Filtered ' if is_filtered else ''
        degree_label = cw.to_ordinal(degree) + ' Degree'
        label = f'{degree_label} Reggresion Fit on {filtered_label}{standarised_label}data'
        coeffs, score = cw.fit_n_degree_regression(degree,
                                                   is_standarised,
                                                   is_filtered)
        print(f'\n{label}:')
        print(f'Coefficients: {coeffs}, R^2: {score:.2f}')
        cw.plot_n_degree_regression(degree, is_standarised, is_filtered)

    # Q2a: Fit a linear regression model to the data
    __sumarise_and_plot_regression_fit(degree=1,
                                       is_standarised=False,
                                       is_filtered=False)

    # 2b: Fit a quadratic regression model to the data
    __sumarise_and_plot_regression_fit(degree=2,
                                       is_standarised=False,
                                       is_filtered=False)

    # 2d: Compare the Akaike Information Criterion (AIC)
    #     for different models
    # NOTE: Question 2d is answered first as
    #       it is used to determine the best model for 2c.
    is_standarised = True
    is_filtered = False
    aic_dict = {k: cw.get_aic(k, is_standarised, is_filtered)
                for k in range(1, 10)}
    d_aic_dict = {k: aic_dict[k] - aic_dict[k - 1] if k > 1 else 0
                  for k in aic_dict}
    print('\n AIC for different regression degrees:')
    print(aic_dict)
    print('\n ΔAIC for different regression degrees:')
    print(d_aic_dict)

    # 2c: Fit the best regression model based on the AIC criterion to
    #     the standardised data
    # NOTE: The best model is the one with the lowest ΔAIC.
    # Set this as the degree of the model:
    degree_2c = min(d_aic_dict, key=d_aic_dict.get)
    print(f'\nBest model based on AIC criterion (Lowest ΔAIC):\
        k = {degree_2c}')
    __sumarise_and_plot_regression_fit(degree_2c,
                                       is_standarised,
                                       is_filtered)
    cw.plot_all_n_degree_regressions((1, degree_2c),
                                     is_standarised,
                                     is_filtered)

    # 2e: Calculate residuals for model in 2c
    #     and plot a histogram and boxplot of the residuals
    residuals = cw.get_residuals(degree_2c, is_standarised, is_filtered)
    cw.plot_residuals(degree_2c, is_standarised, is_filtered)

    # 2f: Extract the filtered data and fit the model in 2c
    #     to the filtered data
    x_flt_st, y_flt = cw.x_flt_st, cw.y_flt
    __sumarise_and_plot_regression_fit(degree_2c,
                                       is_standarised=True,
                                       is_filtered=True)
    return degree_2c


def main(cid, path):
    cw = CourseWork(cid, path)
    # Q1:
    q1_script(cw)
    # Q2:
    degree = q2_script(cw)


if '__main__' == __name__:
    cid = 1524231
    path = 'Statistics/Coursework/rrn18.csv'
    main(cid, path)
