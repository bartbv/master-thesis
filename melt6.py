def melt4(path_climatedata, path_wshed, T_tr, MF, date_start, date_stop):

#%% import libraries:
    import numpy as np
    import pandas as pd
    from numba import jit
    import re

    #Enable JIT to speedup proces

    @jit(nopython=False)
    def snow_reservoir(s, T, T_tr, MF):
        # This function represents the snow storage, calculating the snow stored on the surface

        # s    = snowfall in mm
        # T    = Temperature mm
        # T_tr = Threshold temperature  mm
        # MF    = Meltfactor mm

        M = np.zeros(s.shape[0])
        Ss = np.empty(s.shape[0])
        M[0] = np.where(T[0] > T_tr, np.nanmin([0, (T[0] - T_tr) * MF]), 0)
        Ss[0] = s[0] - M[0]
        for i in range(1, s.shape[0]):
            M[i] = np.where(T[i] > T_tr, np.nanmin([(T[i] - T_tr) * MF, Ss[(i - 1)]]), 0)
            Ss[i] = Ss[(i - 1)] - M[i] + s[i]

        return (M, Ss)

    T_tr = np.float64(T_tr)
    MF = np.float64(MF)

    # Restructuring the data
    df_clim = pd.read_fwf(path_climatedata, widths=[6, 6, 6, 14, 14, 14], header=None)
    df_clim.columns = ['day', 'month', 'year', 'P', 'T', 'E']
    df_clim['date'] = pd.to_datetime(df_clim[['year', 'month', 'day']])
    df_clim = df_clim.set_index('date')
    df_clim = df_clim[date_start.min():date_stop.max()]
    df_clim['date'] = pd.to_datetime(df_clim[['year', 'month', 'day']])

    df_shed = pd.read_excel(path_wshed, index_col=0, header=0)
    df_shed = df_shed.sort_index(0)
    stat_elev = pd.read_excel('C:\\Users\\Bart\\Desktop\\Thesis\\Data\\elevation_stations.xlsx')
    nr_stat = re.findall('\\d+', path_climatedata)


    dT = 0.006  # temperature decrease per 1 meter elevation
    elv_R = float(stat_elev.loc[(stat_elev['HZBNR'] == int(nr_stat[0]), 'RASTERVALU')])
    df_clim['elev'] = pd.Series()
    df_clim['elev'] = elv_R
    df_shed['elev'] = df_shed['gridcode'] * 250 + 125.0 # Calculation of average elevation for each elevation zone
    elev = df_shed['elev']
    df_clim2 = df_clim.copy()

    #initialize dataframe to store the results of the snowmelt calculation
    df_melt = pd.DataFrame()
    df_melt['date'] = pd.to_datetime(df_clim[['year', 'month', 'day']])
    df_cover = pd.DataFrame()
    df_cover['date'] = pd.to_datetime(df_clim[['year', 'month', 'day']])

    # calculate per elevation zone the temperature and check if snow or rain is precipitating
    for i in elev:
        df_elev = pd.DataFrame()
        df_elev['E'] = df_clim['E']
        df_elev['date'] = df_clim['date']
        df_elev['M'] = pd.Series()
        df_elev['Ss'] = pd.Series()
        df_elev['T'] = df_clim['T'] - dT * (i - elv_R)
        df_elev['elev'] = i
        df_elev['S'] = np.where(df_elev['T'] < T_tr, df_clim['P'], 0)
        df_elev['P'] = np.where(df_elev['T'] > T_tr, df_clim['P'], 0)
        #######################################################################
        # df_elev = precip_correction(int(re.findall('(\d+)\.inp', path_climatedata)[0]), df_elev)
        ############################################################################
        s = df_elev['S'].to_numpy('float64')
        T = df_elev['T'].to_numpy('float64')

        # Running the snow module
        df_elev['M'], df_elev['Ss'] = snow_reservoir(s, T, T_tr, MF)
        df_melt[str(i)] = df_elev['M']
        df_melt[str(i) + '_weight'] = df_melt[str(i)] * df_shed['SHAPE_Area'].loc[(df_shed['gridcode'] == int((i - 125) / 250))].values[0] / df_shed['SHAPE_Area'].sum()
        df_cover[str(i)] = df_elev['Ss']
        df_cover[str(i) + '_weight'] = df_shed['SHAPE_Area'].loc[(df_shed['gridcode'] == int((i - 125) / 250))].values[0] / df_shed['SHAPE_Area'].sum()
        df_cover[str(i) + '_cover'] = np.where(df_cover[str(i)] > 0, 1, 0)
        df_clim2 = df_clim2.append(df_elev, ignore_index=True, verify_integrity=True, sort=False)

    z = 1
    df_melt['total'] = pd.Series(np.zeros(df_melt.shape[0]))
    while z <= df_melt.shape[1] - 2:
        df_melt['total'] += df_melt.iloc[:, z + 1]
        z = z + 2

    z = 1
    df_cover['total'] = pd.Series(np.zeros(df_cover.shape[0]))
    while z <= df_melt.shape[1] - 2:
        df_cover['total'] += df_cover.iloc[:, z + 1]
        z = z + 2

    df_cover['cover'] = np.where(df_cover['total'] > 0, 1, 0)
    df_clim2.drop((df_clim2.head(int((date_stop.min() - date_start.max()).days) + 1).index), inplace=True)
    return (df_melt, df_cover, df_clim2, df_clim)




