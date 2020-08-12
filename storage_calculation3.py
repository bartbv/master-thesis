# In this script the cummulative storage deficit is calculated based on daily climate data.
#
def SD_TOT(P, M, ET):
    # This function represents an "infinite" reservoir, calculating the cummulative storage deficit

    # SD_tot = cummulative storage deficit in mm
    # P      = Effective Precipitation mm
    # M      = Melt mm
    # ET     = Transpiration mm

    import numpy as np
    SD_tot = np.zeros((P.shape[0]))
    # Maximum value for SD = 0
    SD_tot[0] = min(0, (P[0] + M[0] - ET[0]))
    # Loop through length of the timeseries
    for i in range(1, P.shape[0]):
        SD_tot[i] = min(0, (SD_tot[i - 1] + P[i] + M[i] - ET[i]))
    return SD_tot

def interception(E, P, Imax):
    # This function represents the interception reservoir, which is equivalent to the water stored on the canopy
    #
    # Pe      = Effective Precipitation mm
    # Ei      = Interception evaporation mm
    # Si      = Interception storage mm

    import numpy as np
    # Set initial conditions to 0
    Pe = np.zeros(P.shape[0])
    Si = np.zeros(P.shape[0])
    Ei = np.zeros(P.shape[0])
    # assumed is an initial empty storage
    Si[0] = 0
    # Loop through length of the timeseries
    for i in range(1, P.shape[0]):
        Si[i] = Si[i-1] + P[i]
        Pe[i] = np.maximum((Si[i] - Imax), 0)
        Si[i] = Si[i] - Pe[i]
        Ei[i] = np.minimum(E[i], Si[i])
        Si[i] = Si[i] - Ei[i]
    return Pe,Ei,Si

def storage_calculation3(station, MF, T_tr,Imax):

    # this function loads the data in to the model. It is proccesing the data in order to be suitable for the different calculation.
    # The model models the flows between the reservoirs and the storage deficit.
#%% set station int tot string
    station = str(station)
#%% Import libraries
    
    import numpy as np
    import pandas as pd
    from melt6 import melt4
    from small_functions.hydroyear import hydroyear
    from small_functions.assign_wy import assign_wy
    from warnings import filterwarnings
    from small_functions.df_clim_ini import df_clim_ini
    from small_functions.df_q import df_q

    # Disable if not using Thorntwaite equation:
    #   from small_functions.Thornthwaite import Thornthwaite
    # from Precip_correction.precip_correction import precip_correction

    # Ignore non essential warnings
    filterwarnings('ignore')

    # Load station information
    df_clim_ini = df_clim_ini(station)
    stat_info   = pd.read_excel(r"C:\Users\Bart\Desktop\Thesis\stations_area.xlsx", sheet_name=1)
    stat_info   = stat_info[stat_info['station_nr'] == int(station)]

    stat_info.set_index('station_nr', inplace=True)

    # Setting the start and end of the hydrological year
    date_start =  pd.to_datetime(stat_info.start_date.values)
    stop_month  = 9
    stop_year   = df_clim_ini.index.year[-1]

    date_stop = pd.to_datetime(stat_info.stop_date.values)

    # Loading Daily climate data

    path_climatedata    = r'C:\Users\Bart\Desktop/Thesis/Data/Austrian Catchments/Austrian Catchments/hotspot_pteps/lumpI_' + station + '.inp'
    path_wshed  = r'C:\Users\Bart\Desktop/Thesis/DEM_final_table/table_final_s' + station + '_DEM_polygon.xlsx'
    pd_shed     = pd.read_excel(path_wshed)  # Inladen watershed data met als doel het oppervlakte te kunnen oproepenp
    # Imax        = 1.5                                                                                                   # change
#%% Calculating snow melt
    # The snow melt from the snow storage reservoir is calculated using the Melft factor and Threshold temperature
    # retreived from the calibration.
    df_melt, df_cover, df_clim2, df_clim = melt4(path_climatedata,path_wshed,T_tr,MF,date_start,date_stop)
    start_WY = hydroyear(df_clim) + 1
    elevations = df_clim2['elev'].unique()

    # %% Weight of the elevation zone in the catchment
    weight = np.empty((len(df_cover), len(elevations)))
    df          = df_cover.filter(like='cover')
    df['date']  = df_cover.loc[:, 'date']
    df.set_index('date', drop=False, inplace=True)
    weight[:]   = df_cover.filter(like='weight').head(1).values
    j = 0
#%% Deze modules zijn gebruikt ter evenutuele correctie van evaporation en precipitation:
    # df_clim3 = Thornthwaite(station, df_clim2)
    # df_clim2 = precip_correction(int(station),df_clim2)

#%% Sum the contribution of the different elevation zones
    for i in elevations:
        df_weight       = pd.DataFrame(columns=['P', 'E', 'S', 'date', 'M','T'])
        df_weight       = df_clim2[df_clim2['elev'] == i]
        df_weight['date'] = df_clim2.date
        df_weight['P']  = df_clim2.P * weight[0][j]
        df_weight['S']  = df_clim2.S * weight[0][j]
        df_weight['M']  = df_clim2.M * weight[0][j]
        df_weight['E']  = df_clim2.E * weight[0][j]
        df_weight['T']  = df_clim2['T'] * weight[0][j]
        df_weight       = df_weight.set_index('date', drop=False)
        df_weight       = df_weight.drop(['elev', 'day', 'month', 'year'], axis=1)
        # df_weight = df_weight.drop(['CE'], axis=1)
        if j == 0:
            df_tot = df_weight.copy()
            df_tot = df_tot.drop(['date'], axis=1)
        else:
            df_weight = df_weight.drop(['date'], axis=1)

            df_tot += df_weight
        j += 1
    df_tot['date'] = df_tot.index.values
    # %% In dit gedeelte wordt het gemiddelde van de verschillende facetten van de waterbalans opgezet. Met als doel de
    #    transpiratie op de lange termijn te berekenen

    df_mean     = pd.DataFrame(columns=['P', 'Q', 'E', 'M', 'ET'])
    df_mean['P']= pd.Series(df_tot.loc[:, 'P'].mean())

    df_mean['M']= pd.Series(df_tot.loc[:, 'M'].mean())

    df_mean_ini     = pd.DataFrame(columns=['P', 'Q', 'E', 'M', 'ET'])
    df_mean_ini['P']= pd.Series(df_clim_ini.loc[:, 'P'].mean())
    df_mean_ini['E']= pd.Series(df_clim_ini.loc[:, 'E'].mean())


    # %% Loading discharge

    df_Q,df_Q_ini   = df_q(station,date_start.strftime("%d-%m-%Y"),date_stop.strftime("%d-%m-%Y"))
    df_tot['Q']     = df_Q['Q'].copy()
    df_mean['Q']    = df_Q['Q'].mean()
    df_mean_ini['Q']= df_Q_ini['Q'].mean()

    # calculation of transpiration scaled with longterm avarage fraction
    lt_wb_ini = 100*(df_mean_ini['P']  - df_mean_ini['E'] - df_mean_ini['Q']) / df_mean_ini['P']

    # convert pandas to numpy for increase in performance:

    data = np.zeros(df_tot.shape[0], dtype={'names': ('WY', 'P', 'M', 'ET', 'E', 'Q', 'SD_tot'),
                                            'formats': ('i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')})
    data['P'] = P = df_tot['P'].to_numpy('float64')
    data['M'] = M = df_tot['M'].to_numpy('float64')
    data['E'] = E = df_tot['E'].to_numpy('float64')
    data['Q'] = Q = df_tot['Q'].to_numpy('float64')


    # Run interception reservoir storage
    df_tot['Pe'],df_tot['Ei'],df_tot['Si']     = interception(E,P,Imax)
    Pe = df_tot['Pe'].to_numpy('float64')

    df_mean['Pe']= pd.Series(df_tot.loc[:, 'Pe'].mean())
    df_mean['Ei']= pd.Series(df_tot.loc[:, 'Ei'].mean())
    df_mean['E'] = pd.Series(df_tot.loc[:, 'E'].mean())
    # Correct the energy balance: substract evaporated water from the interception rservoir from the potential evaporation.
    df_mean['E'] = df_mean['E'] - df_mean['Ei']
    df_tot['E']  = df_tot['E'] - df_tot['Ei']   # remove the intercepation evaporation from potential evaporation of the rootzone storage

    #Long term water balances:
    df_mean['ET']= df_mean['Pe'] + df_mean['M'] - df_mean['Q']

    df_tot['ET'] = (df_tot['E'] * (df_mean['ET'].loc[0] / df_mean['E'].loc[0]))

    data['ET'] = ET = df_tot['ET'].to_numpy('float64')


    factor = [1]
    ET_range = np.zeros((len(df_tot),len(factor)))
    SD_range = np.zeros((len(df_tot),len(factor)))

    # loop not neccesary (factor is a constant)
    for i in range(len(factor)):
        df_tot['date']= pd.to_datetime(df_tot['date'])
        df_tot.index = pd.to_datetime(df_tot.index, format='%Y%m%d')
        #
        df_tot['date'] = df_tot.index
        # assign water year
        df_tot['WY'] = df_tot.apply(lambda x: assign_wy(x, start_WY), axis=1)



        df_tot['SD_tot'] = SD_TOT(Pe, M, ET)
        df_tot['P_tot'] = df_tot['P'] + df_tot['S']
        df_tot['test_Pe1']   = df_tot['P'] - (df_tot['Pe'] + df_tot['Ei']  )
        df_tot['test_Pe2'] = df_tot['P'] - (df_tot['Pe'] + df_tot['Ei'] +  df_tot['Si'] )
        df2 = 1
    # meandata(str(station), 'Q1',date_start,date_stop)  # sanity check

    return df2, df_tot, df_clim2, df_clim, pd_shed, SD_range, ET_range, lt_wb_ini, df_mean

#%% import libraries
import pickle
import pandas as pd
# %% import libraries
import pickle

import pandas as pd

#%% load station numbers
stationlist        = pd.read_pickle('used_stations_list_02_05_2020')
stationlist.index  = stationlist.iloc[:,0]

#%% folders to store output:
###########################
map        = 'Run_08_05'                #change
file       = '08_05.p'            #change
###########################
#%% Load Meltfactor and threshold temperature:

MF_list   = pickle.load( open(r'Data/MF_list_60_new.p '     , "rb" ))
T_tr_list = pickle.load( open(r'Data/T_Tr_list_60_new.p '     , "rb" ))

#%%
calibratie = pd.DataFrame()#volgorde,columns=['station'])
calibratie['MF'] = MF_list
calibratie['T_tr'] = T_tr_list

# calibratie.index = volgorde
#%% load station
stationlist        = pd.read_pickle('used_stations_list_02_05_2020')
stationlist.index  = stationlist.iloc[:,0]
stationlist     = stationlist['station'].values.tolist()
#%%

# for station in station_list.iloc[:,0].values.tolist():
feasible=[]
Ea=[]
Ep=[]
for station in stationlist:

    for Imax in [1.5]:# [0.5,1,2,2.5]:

        MF  = calibratie.loc[station,'MF']
        T_tr= calibratie.loc[station, 'T_tr']
        sd  = storage_calculation3(station, MF,T_tr,Imax)

        #%% this part is to test of in budyko or not:
        def f_budyko(Ept_div_P, Eact_div_P_data):

            if Ept_div_P <= 1:
                Eact_div_P = Ept_div_P
            else:
                Eact_div_P = 1

            if Eact_div_P_data <= Eact_div_P:
                return 'feasible'
            if Eact_div_P_data > Eact_div_P:
                return 'not-feasible'


        df_mean = sd[-1]
        EaP = (1 - (df_mean['Q'][0] / (df_mean['P'][0] + df_mean['M'][0])))
        Ea.append(EaP)
        EpP = ((df_mean['Ei'][0] + df_mean['E'][0]) / (df_mean['P'][0] + df_mean['M'][0]))
        Ep.append(EpP)
        feasible.append(f_budyko(EpP, EaP))


    #%% OUTPUT
        with open("Calculation_Run\\" + map+ '\\' +str(station) +'sd_'  + str(Imax) +'_' + file , 'wb') as f:
            pickle.dump(sd, f)

with open("Calculation_Run\\" + map +'\\' +'calibratie_'+file, 'wb') as f:
    pickle.dump(calibratie, f)

# #%% Checking waterbalances (not neccesary step)
# ###########################
# map        = 'Run_08_05'                #change
# file       = '08_05.p'
# Imax = 1.5
# # runcell(2, 'C:/Users/Bart/PycharmProjects/untitled4/storage_calculation3.py')
# # runcell('load station', 'C:/Users/Bart/PycharmProjects/untitled4/storage_calculation3.py')
# df_tot_tot = pd.DataFrame(columns=['P', 'T', 'E', 'M', 'Ss', 'S', 'date', 'Q', 'Pe', 'Ei', 'Si', 'ET',
#        'WY', 'SD_tot', 'P_tot','station'])
#
# for station in stationlist:
#     station = str(station)
#     sd = pickle.load(open("Calculation_Run\\" + map+ '\\' +str(station) +'sd_'  + str(Imax) +'_' + file , 'rb'))
#     df_tot = sd[1]
#     df_tot['station'] = station
#     df_tot['test_snow'] = df_tot['M'].sum(axis=0) - df_tot['S'].sum(axis=0)
#     df_tot['test_ptot'] = df_tot['P_tot'].sum(axis=0) -( df_tot['P'].sum(axis=0) + df_tot['S'].sum(axis=0))
#     df_tot['test_Pe']   = df_tot['P'].sum(axis=0)- (df_tot['Pe'].sum(axis=0) + df_tot['Ei'].sum(axis=0))
#     df_tot['test_SD']= df_tot['Pe'].sum(axis=0) + df_tot['M'].sum(axis=0) -  df_tot['Q'].sum(axis=0) - df_tot['ET'].sum(axis=0)
#
#     df_tot_tot = df_tot_tot.append(df_tot)
#
# df_tot_tot.to_pickle('df_tot_tot')



#%%

fgdf


