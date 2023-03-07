import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def to_ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


class CourseWorkCode:
    def __init__(self, cid, path):
        self.__set_seed(cid)
        self.x, self.y = self.__read_data(path)
        self.x_st = self.__standardise(self.x)
        self.x_label, self.y_label = 'Time', 'Wavelength (nm)'
        self.data = np.array([self.x, self.y]).T
        self.x_flt, self.y_flt = self.__get_filtered_xy(
            bounds=(10, 850), step=10)
        self.x_flt_st = self.__standardise(self.x_flt)

    def __set_seed(self, cid):
        np.random.seed(cid)

    def __read_data(self, path):
        df = pd.read_csv(path)
        x, y = df['X'].to_numpy(), df['Y'].to_numpy()
        return x, y

    def __standardise(self, x):
        return (x - np.mean(x)) / np.std(x)

    def __get_filtered_xy(self, bounds, step):
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

    def __create_scatter_plot(self, standardise: bool, filter: bool):
        x, y = self.__get_data(standardise, filter)
        filtered_label = 'Filtered ' if filter else ''
        standarised_label = f' with Standardised {self.x_label}' if standardise else ''
        fig, ax = plt.subplots()
        ax.scatter(x,
                   y,
                   s=1,
                   label=filtered_label + 'Data' + standarised_label)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        return ax

    def plot_hist_box(self, bins=10):
        fig, ax = plt.subplots(2, sharex=True, gridspec_kw={
                               'height_ratios': [1, 3]})
        # # Create a boxplot of the y values
        ax[0].boxplot(self.y, vert=False)
        ax[0].set_title('(a)')
        # Create a histogram of the x values
        ax[1].hist(self.y, bins=bins, fc='lightblue', ec='k')
        ax[1].set_xlabel(self.y_label)
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('(b)')
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
        self.__create_scatter_plot(standardise, filter)
        plt.show()

    def fit_n_degree_regression(self, degree: int, standardise: bool, filter: bool):
        x, y = self.__get_data(standardise, filter)
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression()
        model.fit(x_poly, y)
        coeffs = model.coef_
        return model.coef_[1:], model.intercept_, model.score(x_poly, y)

    def plot_n_degree_regression(self, degree: int, standardise: bool, filter: bool):
        x, y = self.__get_data(standardise, filter)
        coeffs, intercept, score = self.fit_n_degree_regression(
            degree, standardise, filter)
        x_fit = np.linspace(x.min(), x.max(), 100)
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        x_fit_poly = poly_features.fit_transform(x_fit.reshape(-1, 1))
        y_fit = np.dot(x_fit_poly, coeffs) + intercept
        ax = self.__create_scatter_plot(standardise, filter)
        ax.plot(x_fit, y_fit, 'r',
                label=f'{to_ordinal(degree)} Degree Fit')
        ax.legend()
        plt.show()

    def plot_all_fits_to_n_degree(self, degree: int, standardise: bool, filter: bool):
        x, y = self.__get_data(standardise, filter)
        ax = self.__create_scatter_plot(standardise, filter)
        for k in range(1, degree+1):
            coeffs, intercept, score = self.fit_n_degree_regression(
                k, standardise, filter)
            x_fit = np.linspace(x.min(), x.max(), 100)
            poly_features = PolynomialFeatures(
                degree=k, include_bias=False)
            x_fit_poly = poly_features.fit_transform(x_fit.reshape(-1, 1))
            y_fit = np.dot(x_fit_poly, coeffs) + intercept
            ax.plot(x_fit, y_fit,
                    label=f'{to_ordinal(k)} Degree Fit')
        ax.legend()
        plt.show()

    def log_likehood(self, degree: int, standardise: bool, filter: bool):
        x, y = self.__get_data(standardise, filter)
        n = len(y)
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly = poly_features.fit_transform(x.reshape(-1, 1))
        # Append a column of ones to the x matrix
        x_poly = np.hstack((np.ones((x_poly.shape[0], 1)), x_poly))
        coeff, intercept, score = self.fit_n_degree_regression(
            degree, standardise, filter)
        # Create the beta vector with intercept as beta_0
        beta = np.hstack((intercept, coeff))
        error = y - np.matmul(x_poly, beta)
        mean_squared_error = np.matmul(error.T, error)
        variance = np.std(y)**2
        exponent = (-1 / (2 * variance)) * mean_squared_error
        log_likehood = -0.5 * n * np.log(2 * np.pi * variance) + exponent

    # def plot_log_likehood(self, degree, standarise=True):


cid = 1524231
path = 'Statistics/Coursework/rrn18.csv'
cw = CourseWorkCode(cid, path)
# Q1a: Plot a histogram and boxplot of the x and y values
cw.plot_hist_box(bins=10)
# Q1b: Summarise the location of the data
print(cw.summarise_location())
# Q1c: Plot a scatter plot of the data
cw.plot_scatter(standardise=False, filter=False)

# Q2a: Fit a linear regression model to the data
n = 1
is_standarised = False
is_filtered = False
coeffs, intercept, score = cw.fit_n_degree_regression(n,
                                                      is_standarised,
                                                      is_filtered)
print('Linear 2 regression fit:')
print(f'Coefficients: {coeffs}, Intercept: {intercept:.2f}, R^2: {score:.2f}')
cw.plot_n_degree_regression(n, is_standarised, is_filtered)

# 2b: Fit a quadratic regression model to the data
n = 2
is_standarised = False
is_filtered = False
coeffs, intercept, score = cw.fit_n_degree_regression(n,
                                                      is_standarised,
                                                      is_filtered)
print('Quadratic regression fit:')
print(f'Coefficients: {coeffs}, Intercept: {intercept:.2f}, R^2: {score:.2f}')
cw.plot_n_degree_regression(n, is_standarised, is_filtered)

# 2c: Fit a nth degree polynomial regression model to the standardised data
n_2c = 7
is_standarised = True
is_filtered = False
coeffs, intercept, score = cw.fit_n_degree_regression(n_2c,
                                                      is_standarised,
                                                      is_filtered)
print(f'{to_ordinal(n_2c)} degree polynomial regression fit:')
print(f'Coefficients: {coeffs}, Intercept: {intercept:.2f}, R^2: {score:.2f}')
cw.plot_n_degree_regression(n_2c, is_standarised, is_filtered)
cw.plot_all_fits_to_n_degree(n_2c, is_standarised, is_filtered)

# 2d: Compare the Akaike Information Criterion (AIC) for different models
cw.log_likehood(2, standardise=True, filter=False)


# 2f: Extract the filtered data and fit the model in 2c to the filtered data
x_flt_st, y_flt = cw.x_flt_st, cw.y_flt
is_standarised = True
is_filtered = True
coeffs, intercept, score = cw.fit_n_degree_regression(n_2c,
                                                      is_standarised,
                                                      is_filtered)
print(f'{to_ordinal(n_2c)} degree polynomial regression fit on filtered data:')
print(f'Coefficients: {coeffs}, Intercept: {intercept:.2f}, R^2: {score:.2f}')
cw.plot_n_degree_regression(n_2c, is_standarised, is_filtered)
