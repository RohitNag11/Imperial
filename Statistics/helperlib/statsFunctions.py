def summarise_data(data, name='Data'):
    print(f'{name} summary:')

    location_dict = {
        'mean': data.mean(),
        'median': data.median(),
        'lowerq': data.quantile(0.25),
        'upperq': data.quantile(0.75),
        'min': min(data),
        'max': max(data)
    }

    dispersion_dict = {
        'range': location_dict['max'] - location_dict['min'],
        'iqr': location_dict['upperq'] - location_dict['lowerq'],
        'var': data.var(),
        'std': data.std()
    }

    skewness_dict = {
        'skew': data.skew(),
        'kurtosis': data.kurt()
    }

    summary_dict = {
        'location': location_dict,
        'dispersion': dispersion_dict,
        'skewness': skewness_dict
    }

    for key in summary_dict:
        print(f' {key}: ')
        print(f'  {summary_dict[key]}')
