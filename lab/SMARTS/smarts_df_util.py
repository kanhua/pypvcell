import numpy as np
import pandas as pd
import os
from SMARTS.smarts import get_clear_sky
from pvlib.tracking import singleaxis


def load_smarts_df(h5_data_files, add_dhi=True):
    """
    Load SMARTS generated DataFrame from different files and combine them into a single DataFrame
    :param h5_data_files:
    :return:
    """

    all_df = []
    for data_file in h5_data_files:
        df = pd.read_hdf(data_file)
        all_df.append(df)

    df = pd.concat(all_df)

    if add_dhi == True:
        df['DHI'] = df['GLOBL_TILT'] - df['BEAM_NORMAL']

    return df


def integrate_timestamp(df, wavelength_step):
    ndf = df.groupby(df.index).sum()
    ndf = ndf.applymap(lambda x: x * wavelength_step)  # the interval is 2 nm

    return ndf


def integrate_by_day(df, time_interval):
    wvl = pd.unique(df['WVLGTH'])
    if len(wvl) > 1:
        raise ValueError("it seems that this dataframe note integrated at each timestamp yet.\
         Try integrate_timestamp first")

    df['date'] = df.index.date
    date_integral = df.groupby(['date']).sum().applymap(lambda x: x * time_interval)
    return date_integral


def ideal_single_axis_tracker_tilt(azimuth, zenith):
    """
    Calculate the ideal tilt angle of a single axis tracker with its axis oriented towards north-south.
    The calculation uses Equation (1) in Ref.1
    We follow the convetion of azimuth angle in SMARTS model:
    north (0 deg), east(90 deg), south (180 deg), east(270 deg)
    Also note that Equation (1) in Ref.1 uses elevation, whereas we use zenith angle here


    Reference:
    [1] Lorenzo, Narvarte, and Muñoz, “Tracking and back‐tracking,”
    Prog Photovoltaics Res Appl, vol. 19, no. 6, pp. 747–753, 2011.


    :param azimuth: azimuth angle in degree. array-like.
    :param zenith: zenith angle in degree. array-like.
    :return: tile angle of the tracker, azimuthal angle of the tracker
    """

    tracker_tilt = np.arctan(
        np.tan(zenith / 360 * 2 * np.pi) * np.sin((-azimuth - 180) / 360 * 2 * np.pi))

    tracker_tilt = tracker_tilt / (2 * np.pi) * 360

    tracker_azim = (tracker_tilt <= 0) * 180 + 90

    return np.abs(tracker_tilt), tracker_azim


def single_axis_traker_angle(out_df: pd.DataFrame):

    n_out_df=out_df.copy()
    tilt, azim=ideal_single_axis_tracker_tilt(out_df['azimuth'].values, out_df['zenith'].values)

    n_out_df['WAZIM'] = azim
    n_out_df['TILT'] = tilt

    return n_out_df

def smarts_spectrum_with_single_axis_tracker(time_range,cache_1='cache1.h5',
                                             cache_2='cache2.h5',force_restart=False,
                                             norm_2pass=True):
    """

    Generate spectrum with single axis tracker

    :param time_range: a datetime-like array
    :param cache_1:
    :param cache_2:
    :param force_restart:
    :param norm_2pass:
    :return:
    """

    if force_restart==False and os.path.exists(cache_2)==True:
        print("Load data from cache.")
        df = pd.read_hdf(cache_2, key='spec_df')
        out_df = pd.read_hdf(cache_2, key='param_df')
        return df,out_df


    # Run the first pass to calculate the solar position
    df,out_df=get_clear_sky(time_range,extend_dict={'TILT':-999})
    df.to_hdf(cache_1,key='spec_df',mode='a')
    out_df.to_hdf(cache_1,key='param_df',mode='a')

    tracker_angle = singleaxis(apparent_azimuth=out_df['azimuth'],
                               apparent_zenith=out_df['zenith'],
                               backtrack=False)

    # Add tracker angle into the parameter datatframe
    out_df = pd.concat([out_df, tracker_angle], axis=1)

    # Rename the tracker azimuth and tilt in order to feed them into 2nd pass SMARTS
    out_df = out_df.rename(index=str, columns={"surface_azimuth": "WAZIM", "surface_tilt": "TILT"})

    # Do the second pass to calculate the output spectrum by using single axis tracker

    df,n_out_df=get_clear_sky(time_range,extend_df=out_df[['TILT','WAZIM']])

    # Renormalize the direct normal incidence
    if norm_2pass==True:
        n_out_df['direct_norm_factor'] = n_out_df['direct_tilt'] / n_out_df['direct_normal']
        df = pd.merge(left=df, right=n_out_df, left_index=True, right_index=True)
        df['BEAM_NORMAL'] = df['BEAM_NORMAL'] * df['direct_norm_factor']


    df.to_hdf(cache_2,key='spec_df',mode='a')
    n_out_df.to_hdf(cache_2,key='param_df',mode='a')

    return df,n_out_df
