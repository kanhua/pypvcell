import pandas as pd
import numpy as np

from pvlib.tracking import SingleAxisTracker
from pvlib.irradiance import total_irrad, aoi_projection
from pvlib.location import Location


class NEDOLocation(object):
    """
    A class that handles an hourly METPV-11 data

    properties that can be accessed:
    - self.loc_name : name of the location
    - self.latitude


    """

    def __init__(self, nedo_day_file, custom_year=2016):
        self.main_df = load_nedo_data(nedo_day_file, custom_year)
        with open(nedo_day_file) as f:
            headline = next(f)
        byte_num, loc_name, lat1, lat2, lon1, lon2, height = headline.split(sep=',')
        self.loc_name = loc_name
        self.latitude = float(lat1) + 0.1 * float(lat2)
        self.longitude = float(lon1) + 0.1 * float(lon2)
        self.altitude = float(height)
        self.extract_unstack_hour_data(norm=False)

    def extract_unstack_hour_data(self, norm=False):
        ext_df = extract_hour_data(self.main_df)
        self.hour_df = unstack_nedo_df(ext_df)

        if norm == True:
            self.hour_df = normlize_sunlight(self.hour_df)

        return self.hour_df

    def upsampling(self, freq_str='15T', interp_type='linear'):

        up_df = interp_nedo_hour_data(self.hour_df, freq_str, interp_type)

        return up_df

    def get_DNI(self):
        """
        Extract DNI from METPV-11 data. Daily METPV-11 data records.
        The raw data in METPV-11 is the incidence on the horizontal plane, that is, DNI*cos(d).
        d is the incidence anlge.

        :return: a dataframe that contains the DNI
        """

        ngo = Location(latitude=self.latitude, longitude=self.longitude, altitude=0, tz='Japan')
        solar_pos = ngo.get_solarposition(pd.DatetimeIndex(self.hour_df['avg_time']))

        cosd = aoi_projection(surface_tilt=0, surface_azimuth=0,
                              solar_zenith=solar_pos['apparent_zenith'], solar_azimuth=solar_pos['azimuth'])

        dni_arr = self.hour_df['DHI'] / cosd

        dni_arr = np.maximum(dni_arr, 0)

        return dni_arr

    def tilt_irr(self, surface_tilt=None, surface_azimuth=180, include_solar_pos=False):
        """
        Calculate the irradiances on a tilted surface

        :param surface_tilt: The surface tilt angle (in degree).
        :param surface_azimuth: The azimuth angle of the surface. Default is 180 degrees.
        :param include_solar_pos: whether to include solar position in the output dataframe.
        :return: a dataframe with calculated solar incidence.
        """

        if surface_tilt is None:
            surface_tilt = self.latitude

        ngo = Location(latitude=self.latitude, longitude=self.longitude, altitude=0, tz='Japan')
        solar_pos = ngo.get_solarposition(pd.DatetimeIndex(self.hour_df['avg_time']))

        dni_arr = self.get_DNI()

        irrad_df = total_irrad(surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
                               apparent_zenith=solar_pos['apparent_zenith'],
                               azimuth=solar_pos['azimuth'], dni=dni_arr,
                               ghi=self.hour_df['GHI'], dhi=self.hour_df['dHI'])

        irrad_df['DNI'] = dni_arr

        n_df = pd.concat([self.hour_df, irrad_df], axis=1)

        if include_solar_pos:
            n_df = pd.concat([n_df, solar_pos], axis=1)

        return n_df

    def single_axis_irr(self, include_track_angles=False):
        """
        Calculate the irradiances incident on a surface that mounted on an ideal single-axis tracker

        :param include_track_angles: whether to include the tracker angles into the returned dataframe.
        :return: a dataframe of sun irradiances on the single-axis tracked surface.
        """

        tracker = SingleAxisTracker(axis_tilt=0, axis_azimuth=0, max_angle=180, backtrack=False)

        ngo = Location(latitude=self.latitude, longitude=self.longitude, altitude=0, tz='Japan')
        solar_pos = ngo.get_solarposition(pd.DatetimeIndex(self.hour_df['avg_time']))

        tracker_angle = tracker.singleaxis(apparent_azimuth=solar_pos['azimuth'],
                                           apparent_zenith=solar_pos['apparent_zenith'])

        dni_arr = self.get_DNI()

        irr = tracker.get_irradiance(dni=dni_arr, ghi=self.hour_df['GHI'], dhi=self.hour_df['dHI'],
                                     solar_zenith=solar_pos['apparent_zenith'],
                                     solar_azimuth=solar_pos['azimuth'],
                                     surface_tilt=tracker_angle['surface_tilt'],
                                     surface_azimuth=tracker_angle['surface_azimuth'])

        irr['DNI'] = dni_arr

        n_df = pd.concat([self.hour_df, irr], axis=1)

        if include_track_angles == True:
            n_df = pd.concat([n_df, tracker_angle], axis=1)

        return n_df


def generate_col_names():
    col_names = ['measure_type', 'month', 'day', 'wind_speed']

    hours = map(str, range(1, 25))

    col_names.extend(hours)

    col_names.extend(['max', 'min', 'integral', 'avg', 'day_num'])

    return col_names


def _to_na(x):
    """
    NEDO data uses 8888 to represent NA value.
    This function convert 8888 to numpy.nan type

    :param x:
    :return:
    """
    if x == 8888:
        return np.nan
    else:
        return x


def load_nedo_data(file_path, custom_year=2016):
    """
    Load the NEDO file and convert it to pandas dataframe.
    The columns are: ['measure_type', 'month', 'day', 'wind_speed', '1', '2', '3', '4', '5',
       '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
       '18', '19', '20', '21', '22', '23', '24', 'max', 'min', 'integral',
       'avg', 'day_num', 'year']

    The index of the dataframe is date in DatetimeIndex type.
    :param file_path: file name of the NEDO .csv file.
    :return: a pandas dataframe
    """

    col_names = generate_col_names()
    df = pd.read_csv(file_path, skiprows=1, index_col=None, header=None, names=col_names)
    df = df.applymap(lambda x: _to_na(x))

    # make date as index
    df['year'] = custom_year
    time_s = pd.to_datetime(df.loc[:, ['year', 'month', 'day']])
    df.index = time_s

    return df


def extract_hour_data(df):
    """
    Slice the hourly measured data in the dataframe. Add the hours into the index.
    In the returned dataframe, the index is DatetimeIndex by hour. The columns are ``energy`` and ``measure_type``


    :param df: the dataframe generated by ``load_nedo_data()``
    :return: the processed dataframe
    """

    all_df = []
    cols = df.columns
    assert cols[4] == '1'
    assert cols[27] == '24'
    for i in range(df.shape[0]):
        test_s = df.iloc[i, 4:28]
        test_df = pd.DataFrame(data=test_s.values, index=test_s.index, columns=['energy'])
        test_df['date'] = df.index[i]
        test_df['month'] = df.index[i].month
        test_df['year'] = df.index[i].year
        test_df['day'] = df.index[i].day
        test_df['hour'] = test_s.index.astype(np.int)
        test_df['measure_type'] = df.iloc[i, 0].astype(np.int)
        all_df.append(test_df)

    all_df = pd.concat(all_df)
    ndf = pd.to_datetime(all_df[['month', 'year', 'day', 'hour']], utc=False)
    all_df.index = pd.DatetimeIndex(ndf.values, tz='Japan')
    return all_df[['energy', 'measure_type']]


def unstack_nedo_df(df, use_avg_time=True):
    """
    Further processing the dataframe generated by ``extract_hour_data()``.
    It unstack the ``measure_type`` in rows and put them into columns
    The returned dataframe has datetime as the index, the nine different hourly data as the columns,
    and a column 'avg_time' that shifts the index time back by half an hour


    :param df: dataframe processed yby ``extract_hour_data()``.
    :return: processed dataframe
    """
    col_name = ['GHI', 'DHI', 'dHI', 'sun_time', 'temperature', 'wind_dir', 'wind_speed', 'rain', 'snow']
    s = pd.DataFrame(df['energy'].values, index=[df.index, df['measure_type']], columns=['energy'])
    s = s.unstack()
    s.index.rename('time', inplace=True)

    s.columns = s.columns.get_level_values(-1)
    s.columns = col_name

    s['avg_time'] = s.index - pd.tseries.offsets.Minute(30)
    if use_avg_time:
        s.index = pd.DatetimeIndex(s['avg_time'])

    return s


def normlize_sunlight(df):
    """
    Convert the integrated, measured sun energy (0.01MJ/m^2) into Watt/m^2

    :param df:
    :return:
    """

    # Fill sun_time with some small number
    def do_zero(x):
        if x == 0.0:
            return 0.5
        else:
            return x

    df['sun_time'] = df['sun_time'].map(do_zero)

    # 1e4 is for converting the dimension from 0.01 MJ/m^2 to J/m^2
    df['DHI_n'] = df['DHI'] * 1e4 / (df['sun_time'] * 60 * 6)
    df['dHI_n'] = df['dHI'] * 1e4 / (60 * 60)
    df['GHI_n'] = df['DHI_n'] + df['dHI_n']

    return df


def interp_nedo_hour_data(df, freq_str='15T', interp_type='linear'):
    """
    Interpt the NEDO data by specified frequency and the type of interpolation

    :param df:
    :param freq_str:
    :param interp_type:
    :return:
    """
    ngp = df.resample(freq_str).asfreq()
    ngp = ngp.interpolate(interp_type)

    return ngp


if __name__ == "__main__":
    df = load_nedo_data('hw51106year.csv')

    ndf = df.groupby(['measure_type'])
    ghi = ndf.get_group(1)

    print(ghi.head())
