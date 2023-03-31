from ..basic_display import *
from .metrics import *
import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing


def time_delay_embedding(series: pd.Series, n_lags: int, horizon: int):
    """
    Time delay embedding
    Time series for supervised learning
    :param series: time series as pd.Series
    :param n_lags: number of past values to used as explanatory variables
    :param horizon: how many values to forecast
    :return: pd.DataFrame with reconstructed time series
    """
    assert isinstance(series, pd.Series)

    if series.name is None:
        name = 'Series'
    else:
        name = series.name

    n_lags_iter = list(range(n_lags, -horizon, -1))

    X = [series.shift(i) for i in n_lags_iter]
    X = pd.concat(X, axis=1).dropna()
    X.columns = [f'{name}(t-{j - 1})'
                 if j > 0 else f'{name}(t+{np.abs(j) + 1})'
                 for j in n_lags_iter]

    return X

# ========================================== Time Series Analysis    =====================================================


def plot_tsa(series, title='', lags=None, plot_fig_args={'width': 13, 'height': 4, 'dpi': 100}, sort_by_index=True):
    # Dickey-Fuller test (stationary?) Autocorrelation (ACF), PartialAutocorrelation (PACF)

    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if sort_by_index:
        series = series.sort_index()

    fig = plt.figure(figsize=(
        plot_fig_args['width'], plot_fig_args['height']), dpi=plot_fig_args['dpi'])
    layout = (2, 2)

    # TS, mean, std plot
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    # sns.lineplot(series, la,ax=ts_ax)
    series.plot(ax=ts_ax, label='actual')

    series.expanding().mean().plot(ax=ts_ax, label='mean')
    series.expanding().std().plot(ax=ts_ax, label='std')
    ts_ax.legend()
    ts_ax.set_title(title +
                    ' \nDickey-Fuller: p={0:.5f}'.format(adfuller(series)[1]))

    # (P)ACF plots
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    smt.graphics.plot_acf(series, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(series, lags=lags, ax=pacf_ax)

    plt.tight_layout()
    plt.show()


def plot_seasonal_decompose(series, model='add', period=None, plot_fig_args=None, ticks=None):
    # Seasonal Decomposition

    series = series.sort_index()
    fig = sm.tsa.seasonal_decompose(series, model=model, period=period).plot()

    if plot_fig_args is not None:
        fig.set_dpi(plot_fig_args['dpi'])
        fig.set_size_inches(plot_fig_args['width'], plot_fig_args['height'])

    fig.suptitle(('Additive' if model == 'add' else 'Multiplicative') +
                 ' Seasonal Decomposition (ADF p={0:.5f})'.format(adfuller(series)[1]))
    if ticks is not None:
        ticks = series.index[range(0, len(series), len(series)//10)]
        fig.axes[0].set_xticks(ticks)
    fig.tight_layout()
    fig.show()


def make_stationary(data: pd.Series, alpha: float = 0.05, max_diff_order: int = 10) -> dict:
    # Test to see if the time series is already stationary
    if adfuller(data)[1] < alpha:
        return {
            'differencing_order': 0,
            'time_series': np.array(data)
        }

    # A list to store P-Values
    p_values = []

    # Test for differencing orders from 1 to max_diff_order (included)
    for i in range(1, max_diff_order + 1):
        # Perform ADF test
        result = adfuller(data.diff(i).dropna())
        # Append P-value
        p_values.append((i, result[1]))

    # Keep only those where P-value is lower than significance level
    significant = [p for p in p_values if p[1] < alpha]
    # Sort by the differencing order
    significant = sorted(significant, key=lambda x: x[0])

    # Get the differencing order
    diff_order = significant[0][0]

    # Make the time series stationary
    stationary_series = data.diff(diff_order).dropna()

    return {
        'differencing_order': diff_order,
        'time_series': np.array(stationary_series)
    }

# ===============================================================================================================================


# =========================================== Moving Average ====================================================================


def plot_cumulative_average(series, window, xlabel, ylabel, plot_fig_args={}):
    cumulative_mean = series.expanding().mean()

    ax = lineplot(cumulative_mean, xlabel, ylabel, 'Moving Average',
                  'Cumulative mean trend', 'g', plot_fig_args, True)

    lineplot(series[window:], label='Actual values', ax=ax)


def plot_moving_average(series, window, xlabel, ylabel, plot_intervals=False, scale=1.96, plot_fig_args={}):
    rolling_mean = series.rolling(window=window).mean()

    ax = lineplot(rolling_mean, xlabel, ylabel, 'Moving Average (window size = {})'.format(
        window), 'Rolling mean trend', 'g', plot_fig_args, True)

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)

        lineplot(upper_bound, marker='r--',
                 label='Upper Bound / Lower Bound', ax=ax)
        lineplot(lower_bound, marker='r--', ax=ax)

    lineplot(series[window:], label='Actual values', ax=ax)

# ===============================================================================================================================


# ============================================ Exponential Smoothing =============================================================

def smooth_values(model, forecast_steps=0):
    smoothed_values = model.fittedvalues.sort_index()
    res = {'smoothed_values': smoothed_values}

    if forecast_steps > 0:
        forecasted_values = model.forecast(forecast_steps)
        indexes = smoothed_values.index
        forecasted_values.index = [
            indexes[-1]+(indexes[1]-indexes[0])*(i+1) for i in range(forecast_steps)]
        # return pd.concat([smoothed_values, forecasted_values])
        res['forecasted_values'] = forecasted_values

    return res


# ----------------------------- Simple Exponential Smoothing (SES) ------------------------------------------------

# def exponential_smoothing(series, alpha):
#     result = [series[0]]  # first value is same as series
#     for n in range(1, len(series)):
#         result.append(alpha * series[n] + (1 - alpha) * result[n-1])
#     return result

# def exponential_smoothing(series, alpha):
#     return series.ewm(alpha=alpha).mean()

def simple_exponential_smoothing(series, alpha=None, forecast_steps=0):
    fit = None
    res = {}

    if alpha is not None:
        fit = SimpleExpSmoothing(series, initialization_method='heuristic').fit(
            smoothing_level=alpha, optimized=False
        )
    else:  # Smoothing factor (alpha) auto optimization
        fit = SimpleExpSmoothing(
            series, initialization_method='estimated').fit(optimized=True)
        res['alpha'] = fit.model.params['smoothing_level']

    res.update(smooth_values(fit, forecast_steps))
    return res


def plot_simple_exponential_smoothing(series, xlabel, ylabel, alphas=[], forecast_steps=0, plot_fig_args={}):
    # Plot base values
    plot_title = 'Exponential Smoothing' + ((' with '+str(forecast_steps) +
                                             ' steps forcasting') if forecast_steps > 0 else '')
    ax = lineplot(series, plot_title, xlabel, ylabel,
                  'Actual', '-', plot_fig_args, True)

    # Plot smooth values
    if len(alphas) == 0:  # Smoothing factor (alpha) auto optimization
        ses = simple_exponential_smoothing(
            series, None, forecast_steps)
        alpha = ses['alpha']

        lineplot(ses['smoothed_values'],
                 label='α={} (optimized)'.format(alpha), ax=ax)

        if forecast_steps > 0:  # Plot forecasted values
            lineplot(ses['forecasted_values'], marker='--',
                     label='α={} (forecast)'.format(alpha), ax=ax)
    else:
        for alpha in alphas:
            ses = simple_exponential_smoothing(
                series, alpha, forecast_steps)
            lineplot(ses['smoothed_values'],
                     label='α={}'.format(alpha), ax=ax)

            if forecast_steps > 0:  # Plot forecasted values
                lineplot(ses['forecasted_values'], marker='--',
                         label='α={} (forecast)'.format(alpha), ax=ax)


# ----------------------------- Double Exponential Smoothing (Holt's method) ------------------------------------------------

def double_exponential_smoothing(series, alpha=None, beta=None, phi=None, additive_model=True, forecast_steps=0):
    fit = None
    res = {}

    if alpha is not None and beta is not None:
        if phi is not None:
            fit = Holt(series, initialization_method='heuristic', exponential=not additive_model, damped_trend=True).fit(
                smoothing_level=alpha, smoothing_trend=beta, damping_trend=phi, optimized=False
            )
        else:
            fit = Holt(series, initialization_method='heuristic', exponential=not additive_model).fit(
                smoothing_level=alpha, smoothing_trend=beta, optimized=False
            )
    else:  # Smoothing factors (alpha & beta) auto optimization
        fit = Holt(
            series, initialization_method='estimated', exponential=not additive_model, damped_trend=True).fit(optimized=True)
        res['alpha'] = fit.model.params['smoothing_level']
        res['beta'] = fit.model.params['smoothing_trend']
        res['phi'] = fit.model.params['damping_trend']

    res.update(smooth_values(fit, forecast_steps))
    return res


def plot_double_exponential_smoothing(series, xlabel, ylabel, alphas=[], betas=[], phis=[], additive_model=True, multiplicative_model=False, forecast_steps=0, plot_fig_args={}):
    # Plot base values
    plot_title = 'Double Exponential Smoothing (Holt)' + ((' with '+str(forecast_steps) +
                                                           ' steps forcasting') if forecast_steps > 0 else '')
    ax = lineplot(series, plot_title, xlabel, ylabel,
                  'Actual', '-', plot_fig_args, True)

    # Plot smooth values for each model type (additive or/and multiplicative)
    model_types = []
    if additive_model or (not additive_model and not multiplicative_model):
        model_types.append('Add')
    if multiplicative_model:
        model_types.append('Mult')

    for model_type in model_types:
        if len(alphas) == 0 or len(betas) == 0:
            # Smoothing factors (alpha,  beta, phi) auto optimization
            ses = double_exponential_smoothing(
                series, None, None, None, model_type == 'Add', forecast_steps)
            alpha = ses['alpha']
            beta = ses['beta']
            phi = ses['phi']

            factors_label = ' α={} β={} ϕ={}'.format(
                alpha, beta, phi)

            lineplot(ses['smoothed_values'],
                     label=model_type + ' optim' + factors_label, ax=ax)

            if forecast_steps > 0:  # Plot forecasted values
                lineplot(ses['forecasted_values'], marker='--',
                         label=model_type + ' pred' + factors_label, ax=ax)
        else:
            if len(phis) == 0:  # No damped trend then pass phi=None to the Holt model
                phis.append(None)

            for alpha in alphas:
                for beta in betas:
                    for phi in phis:
                        ses = double_exponential_smoothing(
                            series, alpha, beta, phi, model_type == 'Add', forecast_steps)
                        factors_label = (' α={} β={}'.format(
                            alpha, beta)) + (' ϕ={}'.format(phi) if phi is not None else '')

                        lineplot(ses['smoothed_values'],
                                 label=model_type + factors_label, ax=ax)

                        if forecast_steps > 0:  # Plot forecasted values
                            lineplot(ses['forecasted_values'], marker='--',
                                     label=model_type + ' pred' + factors_label, ax=ax)


# ----------------------------- Simple Exponential Smoothing (Holt's Winters method) -----------------------------------------

def triple_exponential_smoothing(series, seasonal_periods, alpha=None, beta=None, gamma=None, phi=None, additive_trend=True, additive_season=True, use_boxcox=True, forecast_steps=0):
    fit = None
    res = {}

    if alpha is not None and beta is not None:
        if phi is not None:
            fit = ExponentialSmoothing(series, initialization_method='heuristic', seasonal_periods=seasonal_periods, trend='add' if additive_trend else 'mul', seasonal='add' if additive_season else 'mul', use_boxcox=use_boxcox, damped_trend=True).fit(
                smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, damping_trend=phi, optimized=False
            )
        else:
            fit = ExponentialSmoothing(series, initialization_method='heuristic', seasonal_periods=seasonal_periods, trend='add' if additive_trend else 'mul', seasonal='add' if additive_season else 'mul', use_boxcox=use_boxcox).fit(
                smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False
            )
    else:  # Smoothing factors (alpha & beta) auto optimization
        fit = ExponentialSmoothing(
            series, initialization_method='estimated', seasonal_periods=seasonal_periods, trend='add' if additive_trend else 'mul', seasonal='add' if additive_season else 'mul', use_boxcox=use_boxcox, damped_trend=True).fit(optimized=True)
        res['alpha'] = fit.model.params['smoothing_level']
        res['beta'] = fit.model.params['smoothing_trend']
        res['gamma'] = fit.model.params['smoothing_seasonal']
        res['phi'] = fit.model.params['damping_trend']

    res.update(smooth_values(fit, forecast_steps))
    return res


def plot_triple_exponential_smoothing(series, xlabel, ylabel, seasonal_periods, alphas=[], betas=[], gammas=[], phis=[], additive_trend=True, multiplicative_trend=False, additive_season=True, multiplicative_season=False, use_boxcox=True, forecast_steps=0, plot_fig_args={}):
    # Plot base values
    plot_title = 'Tripe Exponential Smoothing (Holt-Winters)' + ((' with '+str(forecast_steps) +
                                                                  ' steps forcasting') if forecast_steps > 0 else '')
    ax = lineplot(series, plot_title, xlabel, ylabel,
                  'Actual', '-', plot_fig_args, True)

    # Plot smooth values for each model type (additive or/and multiplicative)
    trend_types = []
    if additive_trend or (not additive_trend and not multiplicative_trend):
        trend_types.append('Add')
    if multiplicative_trend:
        trend_types.append('Mult')

    season_types = []
    if additive_season or (not additive_season and not multiplicative_season):
        season_types.append('Add')
    if multiplicative_season:
        season_types.append('Mult')

    for seasonal_period in seasonal_periods:
        for trend_type in trend_types:
            for season_type in season_types:
                model_type = trend_type+'-'+season_type

                if len(alphas) == 0 or len(betas) == 0 or len(gammas) == 0:
                    # Smoothing factors (alpha,  beta, phi) auto optimization
                    ses = triple_exponential_smoothing(
                        series, seasonal_period, None, None, None, None, trend_type == 'Add', season_type == 'Add', use_boxcox, forecast_steps)
                    alpha = ses['alpha']
                    beta = ses['beta']
                    gamma = ses['gamma']
                    phi = ses['phi']

                    factors_label = ' sp={} α={} β={} γ={} ϕ={}'.format(
                        seasonal_period, alpha, beta, gamma, phi)

                    lineplot(ses['smoothed_values'],
                             label=model_type + ' optim' + factors_label, ax=ax)

                    if forecast_steps > 0:  # Plot forecasted values
                        lineplot(ses['forecasted_values'], marker='--',
                                 label=model_type + ' pred' + factors_label, ax=ax)
                else:
                    if len(phis) == 0:  # No damped trend then pass phi=None to the Holt model
                        phis.append(None)

                    for alpha in alphas:
                        for beta in betas:
                            for gamma in gammas:
                                for phi in phis:
                                    ses = triple_exponential_smoothing(
                                        series, seasonal_period, alpha, beta, gamma, phi, trend_type == 'Add', season_type == 'Add', use_boxcox, forecast_steps)
                                    factors_label = (' sp={} α={} β={} γ={}'.format(seasonal_period,
                                                                                    alpha, beta, gamma)) + (' ϕ={}'.format(phi) if phi is not None else '')

                                    lineplot(ses['smoothed_values'],
                                             label=model_type + factors_label, ax=ax)

                                    if forecast_steps > 0:  # Plot forecasted values
                                        lineplot(ses['forecasted_values'], marker='--',
                                                 label=model_type + ' pred' + factors_label, ax=ax)

# ===============================================================================================================================


# ========================================== Forecast plot ================= =====================================================

def plot_forecast(series, predictions, in_sample_predictions, model_name, horizon=1, error_name: list = ['MAPE'], error_callback=[mean_absolute_percentage_error], title='', xlabel=None, ylabel=None, plot_fig_args={}, ax=None, ticks=None):
    # transform prediction into Series if it is not
    if isinstance(predictions, pd.DataFrame):
        predictions = pd.Series(predictions.iloc[:, 0])
    if not isinstance(predictions, pd.Series):
        predictions = pd.Series(predictions)

    # Balance in_sample_predictions and predictions size
    series_size = len(series)
    predictions_size = len(predictions)
    if series_size < predictions_size:
        in_sample_predictions = series_size
        predictions = predictions[-in_sample_predictions:]

    # Compute the predictions index
    index = series.index
    predictions_index = list(index[-in_sample_predictions:])
    predictions_index += [
        index[-1]+(index[1]-index[0])*(i+1) for i in range(len(predictions)-in_sample_predictions)]
    # Create the Series
    predictions = pd.Series(predictions.values, index=predictions_index)

    # Compute the prediction errors following the given error metrics
    error_text = '\n'
    for j in range(len(error_name)):
        error = error_callback[j](
            series[-in_sample_predictions:], predictions[:in_sample_predictions])
        error_text += error_name[j]+'={0:.3f}'.format(error)+' '

    # Display the base data
    ax = lineplot(series, 'Forecasting '+title+error_text, xlabel, ylabel,
                  'Actual', '-', plot_fig_args, True, ax, ticks)
    # Display the predictions
    lineplot(predictions, label=model_name+' prediction',
             marker='--', plot_fig_args=plot_fig_args, ax=ax)


def plot_multi_forecast(series, all_predictions: list, all_in_sample_predictions: list, model_names: list, error_name: list, error_callback: list, title='', xlabel=None, ylabel=None, plot_fig_args={}, ax=None, ticks=None):
    # Check parameters sizes
    if not (len(all_predictions) == len(all_in_sample_predictions) == len(model_names)):
        raise Exception(
            'The list type prediction parameters (predictions, in_sample_predictions, model_name) must have the same size')
    if not (len(error_name) == len(error_callback)):
        raise Exception(
            'The list type error parameters (error_name, error_callback) must have the same size')

    # Display the base data
    ax = lineplot(series, 'Forecasting '+title, xlabel,
                  ylabel, 'Actual', '-', plot_fig_args, True, ax=ax, ticks=ticks)
    # Display all the predictions
    for i in range(len(all_predictions)):
        in_sample_predictions = all_in_sample_predictions[i]
        predictions = all_predictions[i]
        # transform prediction into Series if it is not
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions)

        # Compute the predictions index
        index = series.index
        predictions_index = list(index[-in_sample_predictions:])
        predictions_index += [
            index[-1]+(index[1]-index[0])*(i+1) for i in range(len(predictions)-in_sample_predictions)]
        # Create the Series
        predictions = pd.Series(
            predictions.values, index=predictions_index)

        # Compute the prediction errors following the given error metrics
        label = model_names[i]+' prediction'
        for j in range(len(error_name)):
            error = error_callback[j](
                series[-in_sample_predictions:], predictions[:in_sample_predictions])
            label += ' '+error_name[j]+'={0:.3f}'.format(error)
        label += ')'

        lineplot(predictions, label=label, marker='--',
                 plot_fig_args=plot_fig_args, ax=ax)

# ===============================================================================================================================
