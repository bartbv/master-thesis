#%% Gumbel extreme value distribution
def gumbel(df_tot):
    # this function is calculating the
    import numpy as np



    # inverse sign
    df_tot['SD_tot'] = df_tot['SD_tot'] * -1

    # select maximum values for each water year
    a           = df_tot.groupby('WY').max()

    # select minimum values for each water year
    b           = df_tot.groupby('WY').min()

    a           = a - b


    a = a.sort_values('SD_tot', ascending=False)
    # rank yeas from large to small
    a['rank']   = range(len(a))
    a['rank']   = a['rank'].add(1)
    a['p']      = a['rank'] / (len(a) + 1)
    a['T']      = 1 / a['p']
    a['logT']   = np.log(a['T'])
    a['T']      = 1 / a['p']
    a['Tyear']  = a['T'] / 365

    # calculate the reduced variate
    a['y']      = -np.log(-np.log(a['p']))
    a['sy']     = np.std(a['y'])
    a['ym']     = np.mean(a['y'])
    a['xm']     = np.mean(a['SD_tot'])
    a['s']      = np.std(a['SD_tot'])
    a['a']      = a['sy'] / a['s']
    a['b']      = a['xm'] - a['s'] * (a['ym'] / a['sy'])
    a['gum']    = a['a'] * (a['SD_tot'] - a['b'])

    # for different return periods
    y2          = -np.log(-np.log(1 - (1 / 2)))
    y5          = -np.log(-np.log(1 - (1 / 5)))
    y10         = -np.log(-np.log(1 - (1 / 10)))
    y20         = -np.log(-np.log(1 - (1 / 20)))
    y50         = -np.log(-np.log(1 - (1 / 50)))

    sr2         = (y2 / a['a'].iloc[0]) + a['b'].iloc[0]
    sr5         = (y5 / a['a'].iloc[0]) + a['b'].iloc[0]
    sr10        = (y10 / a['a'].iloc[0]) + a['b'].iloc[0]
    sr20        = (y20 / a['a'].iloc[0]) + a['b'].iloc[0]


    return [sr2, sr5, sr10, sr20]