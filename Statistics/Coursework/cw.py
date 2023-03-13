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

    def __fit_n_degree_regression_to_data(self, x, y, degree):
        x_poly = self.__get_x_poly(x, degree)
        model = LinearRegression()
        model.fit(x_poly, y)
        score = model.score(x_poly, y)
        beta = model.coef_
        beta[0] = model.intercept_
        return beta, score

    def fit_n_degree_regression(self,
                                degree: int,
                                standardise: bool,
                                filter: bool):
        x, y = self.__get_data(standardise, filter)
        beta, score = self.__fit_n_degree_regression_to_data(x, y, degree)
        return beta, score

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
        fig, ax = plt.subplots(1, 3, sharey=True, gridspec_kw={
                               'width_ratios': [1, 1, 1]})
        # Plot the scatter plot of errors on the first subplot
        ax[0].scatter(x, error, s=1, label='Residuals')
        ax[0].axhline(mean_error, c='r', ls='--', label='Mean Error')
        self.__config_plot(ax[0],
                           x_label=x_label,
                           y_label='Error (nm)',
                           title='(a)',
                           legend=True)
        # Plot the horizontal histogram of errors on the second subplot
        ax[1].hist(error, bins=10, fc='lightblue',
                   ec='k', orientation='horizontal')
        ax[1].axhline(mean_error, c='r', ls='--', label='Mean Error')
        self.__config_plot(ax[1],
                           x_label='Frequency',
                           y_label='Error (nm)',
                           title='(b)',
                           legend=False)
        # Plot the QQ plot of errors on the third subplot
        stats.probplot(error, rvalue=True, plot=ax[2])
        self.__config_plot(ax[2],
                           x_label='Theoretical Quantiles',
                           y_label='Sample Quantiles',
                           title='(c)')
        # plt.subplots_adjust(wspace=0.3)
        plt.show()

    def __log_likehood(self, degree: int, standardise: bool, filter: bool):
        # x, y = self.__get_data(standardise, filter)
        error = self.get_residuals(degree, standardise, filter)
        n = len(error)
        squared_error = np.matmul(error.T, error)
        sigma_hat_squared = np.std(error)**2
        # sigma_hat_squared = sum(error**2) / n
        exponent = (-1 / (2 * sigma_hat_squared)) * squared_error
        log_likehood = -0.5 * n * \
            np.log(2 * np.pi * sigma_hat_squared) + exponent
        return log_likehood

    def get_aic(self, degree: int, standardise: bool, filter: bool):
        log_likehood = self.__log_likehood(degree, standardise, filter)
        q = degree + 2
        aic = 2 * q - 2 * log_likehood
        return aic

    def __get__bootstrapped_error(self, degree: int, standardise: bool, filter: bool):
        error = self.get_residuals(degree, standardise, filter)
        return np.random.choice(error,
                                size=len(error),
                                replace=True)

    def __get_response_from_bootstrapped_error(self,
                                               degree: int,
                                               standardise: bool,
                                               filter: bool):
        x, y = self.__get_data(standardise, filter)
        x_poly = self.__get_x_poly(x, degree)
        beta, score = self.fit_n_degree_regression(degree,
                                                   standardise,
                                                   filter)
        error = self.__get__bootstrapped_error(degree,
                                               standardise,
                                               filter)
        return np.dot(x_poly, beta) + error

    def __fit_n_degree_regression_to_bootstrapped_response(self,
                                                           degree: int,
                                                           standardise: bool,
                                                           filter: bool):
        x, y = self.__get_data(standardise, filter)
        x_poly = self.__get_x_poly(x, degree)
        y_bootstrapped = self.__get_response_from_bootstrapped_error(
            degree, standardise, filter)
        beta, score = self.__fit_n_degree_regression_to_data(
            x, y_bootstrapped, degree)
        y_bootstrapped_fit = np.dot(x_poly, beta)
        return y_bootstrapped_fit

    def get_response_confidence_band(self,
                                     repeat: int,
                                     confidence_interval: float,
                                     degree: int,
                                     standardise: bool,
                                     filter: bool):
        def __get_confidence_band(array):
            lower = np.percentile(array, (100 - confidence_interval) / 2)
            upper = np.percentile(array, (100 + confidence_interval) / 2)
            return lower, upper
        Y = np.array([self.__fit_n_degree_regression_to_bootstrapped_response(
            degree, standardise, filter) for _ in range(repeat)])
        confidence_band = np.apply_along_axis(__get_confidence_band,
                                              axis=0,
                                              arr=Y)
        return confidence_band

    def __create_confidence_band_on_plot(self,
                                         ax,
                                         repeat: int,
                                         confidence_interval: float,
                                         degree: int,
                                         standardise: bool,
                                         filter: bool):
        x, y = self.__get_data(standardise, filter)
        confidence_band = self.get_response_confidence_band(
            repeat, confidence_interval, degree, standardise, filter)
        ax.fill_between(x,
                        confidence_band[0],
                        confidence_band[1],
                        alpha=0.1,
                        label=f'{confidence_interval}% Confidence Band for {repeat} Bootstrap Samples')
        # ax.plot(x, confidence_band[0], c='r', ls='--', lw=1)
        # ax.plot(x, confidence_band[1], c='r', ls='--', lw=1)
        return ax

    def plot_confidence_band(self,
                             repeat: int,
                             confidence_interval: float,
                             degree: int,
                             standardise: bool,
                             filter: bool):
        x, y = self.__get_data(standardise, filter)
        x_fit = np.linspace(np.min(x), np.max(x), 100)
        x_fit_poly = self.__get_x_poly(x_fit, degree)
        y_fit_original = np.dot(x_fit_poly,
                                self.fit_n_degree_regression(degree,
                                                             standardise,
                                                             filter)[0])
        x_label = f'Standardised {self.x_label}' \
            if standardise else self.x_label
        ax = self.__create_scatter_plot(standardise, filter)
        self.__create_confidence_band_on_plot(ax,
                                              repeat,
                                              confidence_interval,
                                              degree,
                                              standardise,
                                              filter)
        ax.plot(x_fit,
                y_fit_original,
                c='r',
                label=f'{self.to_ordinal(degree)} Degree Fit')
        self.__config_plot(ax, x_label, self.y_label, legend=True)
        plt.show()

    def plot_confidence_band_error_for_dif_samples(self,
                                                   repeat_range: tuple,
                                                   confidence_interval: float,
                                                   degree: int,
                                                   standardise: bool,
                                                   filter: bool):
        x, y = self.__get_data(standardise, filter)
        x_poly = self.__get_x_poly(x, degree)
        y_fit_original = np.dot(x_poly,
                                self.fit_n_degree_regression(degree,
                                                             standardise,
                                                             filter)[0])
        bootstrap_samples = np.arange(
            repeat_range[0], repeat_range[1] + 1, repeat_range[2])
        conf_band_mean_squared_errors = np.zeros(bootstrap_samples.shape[0])
        for i, repeat in enumerate(bootstrap_samples):
            confidence_band = self.get_response_confidence_band(
                repeat, confidence_interval, degree, standardise, filter)
            mean_confidence_band = np.mean(confidence_band, axis=0)
            conf_band_mean_squared_errors[i] = np.mean(
                np.square(mean_confidence_band - y_fit_original))
        fig, ax = plt.subplots()
        ax.plot(bootstrap_samples, conf_band_mean_squared_errors)
        self.__config_plot(ax,
                           'Bootstrap Samples',
                           'Confidence Band MSE',
                           legend=False)
        plt.show()


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
        standarised_label = 'Standarised ' if is_standarised else 'Raw '
        filtered_label = 'Filtered ' if is_filtered else ''
        degree_label = cw.to_ordinal(degree) + ' Degree'
        label = f'{degree_label} Reggresion Fit on {filtered_label}{standarised_label}data'
        coeffs, score = cw.fit_n_degree_regression(degree,
                                                   is_standarised,
                                                   is_filtered)
        print(f'\n{label}:')
        print(f'Coefficients: {coeffs}, R^2: {score}')
        cw.plot_n_degree_regression(degree, is_standarised, is_filtered)

    # Q2a: Fit a linear regression model to the data
    __sumarise_and_plot_regression_fit(degree=1,
                                       is_standarised=False,
                                       is_filtered=False)

    # Q2b: Fit a quadratic regression model to the data
    __sumarise_and_plot_regression_fit(degree=2,
                                       is_standarised=False,
                                       is_filtered=False)
    # Plot the linear and quadratic regression fits on the same plot
    cw.plot_all_n_degree_regressions((1, 2),
                                     standardise=False,
                                     filter=False)

    # Q2d: Compare the Akaike Information Criterion (AIC)
    #     for different models
    # NOTE: Question 2d is answered first as
    #       it is used to determine the best model for 2c.
    degree_range_to_test = (1, 10)
    is_standarised = True
    is_filtered = False
    # Plot all the regression fits
    cw.plot_all_n_degree_regressions(degree_range_to_test,
                                     is_standarised,
                                     is_filtered)
    aic_dict = {k: cw.get_aic(k, is_standarised, is_filtered)
                for k in range(degree_range_to_test[0], degree_range_to_test[1] + 1)}
    print('\n AIC for different regression degrees:')
    print(aic_dict)

    # NOTE: The best model is the one with the lowest AIC.
    #       Set this as the degree of the model:
    degree_2c = min(aic_dict, key=aic_dict.get)
    print(f'\nBest model based on AIC criterion (Lowest AIC):\
        k = {degree_2c}')

    # Q2c: Fit the best regression model based on the AIC criterion to
    #     the standardised data
    __sumarise_and_plot_regression_fit(degree_2c,
                                       is_standarised,
                                       is_filtered)

    # Q2e: Calculate residuals for model in 2c
    #     and plot a scatter plot, histogram and QQ plot of the residuals
    residuals = cw.get_residuals(degree_2c, is_standarised, is_filtered)
    cw.plot_residuals(degree_2c, is_standarised, is_filtered)

    # Q2f: Extract the filtered data and fit the model in 2c
    #     to the filtered data
    x_flt_st, y_flt = cw.x_flt_st, cw.y_flt
    __sumarise_and_plot_regression_fit(degree_2c,
                                       is_standarised=True,
                                       is_filtered=True)
    return degree_2c


def q3_script(cw, degree):
    # Q3ai: Calculate the residuals for the model in 2f
    residuals = cw.get_residuals(degree,
                                 standardise=True,
                                 filter=True)
    # Q3aii: Plot the confidence band for the model in 2f for 1000 samples
    bootstrap_samples = 1000
    cw.plot_confidence_band(repeat=bootstrap_samples,
                            confidence_interval=95,
                            degree=degree,
                            standardise=True,
                            filter=True)
    # Q3b: Show graphically the effect of increasing bootstrap samples
    boostrap_sample_range = (100, 1000, 100)
    cw.plot_confidence_band_error_for_dif_samples(repeat_range=boostrap_sample_range,
                                                  confidence_interval=95,
                                                  degree=degree,
                                                  standardise=True,
                                                  filter=True)


def main(cid, path):
    cw = CourseWork(cid, path)
    # Q1:
    q1_script(cw)
    # Q2:
    degree = q2_script(cw)
    # Q3:
    q3_script(cw, degree)


if '__main__' == __name__:
    cid = 1524231
    path = 'Statistics/Coursework/rrn18.csv'
    main(cid, path)
