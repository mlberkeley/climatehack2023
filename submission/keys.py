from enum import Enum, IntEnum


class META(Enum):
    TIME = 0
    LATITUDE = 1
    LONGITUDE = 2
    ORIENTATION = 3
    TILT = 4
    KWP = 5


class NONHRV(IntEnum):
    IR_016 = 0
    IR_039 = 1
    IR_087 = 2
    IR_097 = 3
    IR_108 = 4
    IR_120 = 5
    IR_134 = 6
    VIS006 = 7
    VIS008 = 8
    WV_062 = 9
    WV_073 = 10


class WEATHER(Enum):
    ALB_RAD = 1
    ASWDIFD_S = 2
    ASWDIR_S = 3
    CAPE_CON = 4
    CLCH = 5
    CLCL = 6
    CLCM = 7
    CLCT = 8
    H_SNOW = 9
    OMEGA_1000 = 10
    OMEGA_700 = 11
    OMEGA_850 = 12
    OMEGA_950 = 13
    PMSL = 14
    RELHUM_2M = 15
    RUNOFF_G = 16
    RUNOFF_S = 17
    T_2M = 18
    T_500 = 19
    T_850 = 20
    T_950 = 21
    T_G = 22
    TD_2M = 23
    TOT_PREC = 24
    U_10M = 25
    U_50 = 26
    U_500 = 27
    U_850 = 28
    U_950 = 29
    V_10M = 30
    V_50 = 31
    V_500 = 32
    V_850 = 33
    V_950 = 34
    VMAX_10M = 35
    W_SNOW = 36
    WW = 37
    Z0 = 38


class AIR_QUALITY(Enum):
    CO_CONC = 0
    DUST = 1
    NH3_CONC = 2
    NMVOC_CONC = 3
    NO2_CONC = 4
    NO_CONC = 5
    O3_CONC = 6
    PANS_CONC = 7
    PM10_CONC = 8
    PM2P5_CONC = 9
    PMWF_CONC = 10
    SIA_CONC = 11
    SO2_CONC = 12


WEATHER_RANGES = {
        WEATHER.ALB_RAD: (0, 100),
        WEATHER.ASWDIFD_S: (0.0, 544.5),
        WEATHER.ASWDIR_S: (0.0, 864.0),
        WEATHER.CAPE_CON: (0.0, 2244.0),
        WEATHER.CLCH: (0.0, 100.0),
        WEATHER.CLCL: (0.0, 100.0),
        WEATHER.CLCM: (0.0, 100.0),
        WEATHER.CLCT: (0.0, 100.0),
        WEATHER.H_SNOW: (0.0, 4.3086),
        WEATHER.OMEGA_1000: (-10.32, 13.172),
        WEATHER.OMEGA_700: (-40.28, 26.891),
        WEATHER.OMEGA_850: (-33.34, 22.797),
        WEATHER.OMEGA_950: (-16.77, 14.117),
        WEATHER.PMSL: (93928.17, 105314.26),
        WEATHER.RELHUM_2M: (0, 100),
        WEATHER.RUNOFF_G: (-0.2157, 153.75),
        WEATHER.RUNOFF_S: (0, 123.94),
        WEATHER.T_2M: (248.9, 313.75),
        WEATHER.T_500: (228.4, 269.75),
        WEATHER.T_850: (250.5, 299.75),
        WEATHER.T_950: (254.1, 309.5),
        WEATHER.T_G: (235.8, 322.5),
        WEATHER.TD_2M: (233.5, 297.75),
        WEATHER.TOT_PREC: (0.0, 150.75),
        WEATHER.U_10M: (-27.22, 29.094),
        WEATHER.U_50: (-28.66, 73.375),
        WEATHER.U_500: (-49.12, 79.5),
        WEATHER.U_850: (-46.31, 47.188),
        WEATHER.U_950: (-37.47, 39.812),
        WEATHER.V_10M: (-26.95, 28.469),
        WEATHER.V_50: (-56.91, 51.812),
        WEATHER.V_500: (-59.09, 60.844),
        WEATHER.V_850: (-37.09, 48.625),
        WEATHER.V_950: (-39.81, 42.469),
        WEATHER.VMAX_10M: (0.05722, 65.062),
        WEATHER.W_SNOW: (0.0, 1422.0),
        WEATHER.WW: (0, 100),
        WEATHER.Z0: (0, 1.0)
}
