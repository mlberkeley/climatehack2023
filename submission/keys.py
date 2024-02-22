from enum import IntEnum, auto, unique

class KeyEnum(IntEnum):
    @classmethod
    def from_str(cls, key: str):
        for k in cls:
            if k.name == key:
                return k
        raise KeyError(f'Key {key} not found in {cls.__name__}')

@unique
class META(KeyEnum):
    TIME = 0
    LATITUDE = auto()
    LONGITUDE = auto()
    ORIENTATION = auto()
    TILT = auto()
    KWP = auto()


@unique
class NONHRV(KeyEnum):
    IR_016 = 0
    IR_039 = auto()
    IR_087 = auto()
    IR_097 = auto()
    IR_108 = auto()
    IR_120 = auto()
    IR_134 = auto()
    VIS006 = auto()
    VIS008 = auto()
    WV_062 = auto()
    WV_073 = auto()


@unique
class WEATHER(KeyEnum):
    ALB_RAD = 0
    ASWDIFD_S = auto()
    ASWDIR_S = auto()
    CAPE_CON = auto()
    CLCH = auto()
    CLCL = auto()
    CLCM = auto()
    CLCT = auto()
    H_SNOW = auto()
    OMEGA_1000 = auto()
    OMEGA_700 = auto()
    OMEGA_850 = auto()
    OMEGA_950 = auto()
    PMSL = auto()
    RELHUM_2M = auto()
    RUNOFF_G = auto()
    RUNOFF_S = auto()
    T_2M = auto()
    T_500 = auto()
    T_850 = auto()
    T_950 = auto()
    T_G = auto()
    TD_2M = auto()
    TOT_PREC = auto()
    U_10M = auto()
    U_50 = auto()
    U_500 = auto()
    U_850 = auto()
    U_950 = auto()
    V_10M = auto()
    V_50 = auto()
    V_500 = auto()
    V_850 = auto()
    V_950 = auto()
    VMAX_10M = auto()
    W_SNOW = auto()
    WW = auto()
    Z0 = auto()


class AIR_QUALITY(KeyEnum):
    CO_CONC = 0
    DUST = auto()
    NH3_CONC = auto()
    NMVOC_CONC = auto()
    NO2_CONC = auto()
    NO_CONC = auto()
    O3_CONC = auto()
    PANS_CONC = auto()
    PM10_CONC = auto()
    PM2P5_CONC = auto()
    PMWF_CONC = auto()
    SIA_CONC = auto()
    SO2_CONC = auto()


# cheats for model training
class FUTURE(KeyEnum):
    NONHRV = 0


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
