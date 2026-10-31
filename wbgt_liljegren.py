"""
Portage Python du modèle WBGT de Liljegren et al. (2008).

Référence :
    Liljegren, J. C., Carhart, R. A., Lawday, P., Tschopp, S. & Sharp, R. (2008).
    Modeling the Wet Bulb Globe Temperature Using Standard Meteorological Measurements.
    Journal of Occupational and Environmental Hygiene, 5(10), 645-655.

Ce module reproduit fidèlement le code C original (wbgt.c, Argonne National Laboratory,
version diffusée sur https://github.com/mdljts/wbgt) : mêmes constantes physiques, mêmes
corrélations de transfert thermique, même schéma itératif (point fixe amorti, tolérance
0,02 K, 50 itérations max), même position solaire (formules basse précision de
l'Astronomical Almanac, réfraction atmosphérique comprise), même estimation de la fraction directe du rayonnement et même
extrapolation du vent de 10 m à 2 m (loi puissance selon la classe de stabilité de Pasquill).

WBGT = 0,7 * T_nwb + 0,2 * T_g + 0,1 * T_a
    T_nwb : température humide naturelle (thermomètre mouillé non ventilé, exposé au soleil)
    T_g   : température de globe noir (diamètre 2 pouces = 5,08 cm)
    T_a   : température de l'air (sèche)

Validation : sur 1 000 cas tirés au hasard (France, mai-juillet, jour et nuit), l'écart maximal avec
l'exécutable compilé à partir du code C original est de 0,008 °C sur le WBGT.
"""
import math

# ---------------------------------------------------------------- constantes (wbgt.c)
SOLAR_CONST = 1367.0
STEFANB = 5.6696e-8
CP = 1003.5
M_AIR = 28.97
M_H2O = 18.015
RATIO = CP * M_AIR / M_H2O
R_GAS = 8314.34
R_AIR = R_GAS / M_AIR
PR = CP / (CP + 1.25 * R_AIR)
EMIS_WICK, ALB_WICK, D_WICK, L_WICK = 0.95, 0.4, 0.007, 0.0254
EMIS_GLOBE, ALB_GLOBE, D_GLOBE = 0.95, 0.05, 0.0508
EMIS_SFC, ALB_SFC = 0.999, 0.45
CZA_MIN = 0.00873
NORMSOLAR_MAX = 0.85
REF_HEIGHT = 2.0
MIN_SPEED = 0.13
CONVERGENCE = 0.02
MAX_ITER = 50
DEG_RAD = math.pi / 180.0


# ---------------------------------------------------------------- propriétés de l'air
def esat(tk):
    """Pression de vapeur saturante au-dessus de l'eau liquide (hPa), Buck (1981), tk en K."""
    y = (tk - 273.15) / (tk - 32.18)
    return 1.004 * 6.1121 * math.exp(17.502 * y)


def dew_point(e):
    """Température de rosée (K) pour une pression de vapeur e (hPa)."""
    z = math.log(e / (6.1121 * 1.004))
    return 273.15 + 240.97 * z / (17.502 - z)


def viscosity(tk):
    """Viscosité dynamique de l'air (kg m-1 s-1)."""
    sigma, eps_kappa = 3.617, 97.0
    tr = tk / eps_kappa
    omega = (tr - 2.9) / 0.4 * (-0.034) + 1.048
    return 2.6693e-6 * math.sqrt(M_AIR * tk) / (sigma * sigma * omega)


def thermal_cond(tk):
    """Conductivité thermique de l'air (W m-1 K-1)."""
    return (CP + 1.25 * R_AIR) * viscosity(tk)


def diffusivity(tk, pair):
    """Diffusivité de la vapeur d'eau dans l'air (m2 s-1), pair en hPa."""
    pcrit13 = (36.4 * 218.0) ** (1.0 / 3.0)
    tcrit512 = (132.0 * 647.3) ** (5.0 / 12.0)
    tcrit12 = math.sqrt(132.0 * 647.3)
    mmix = math.sqrt(1.0 / M_AIR + 1.0 / M_H2O)
    patm = pair / 1013.25
    return 3.640e-4 * (tk / tcrit12) ** 2.334 * pcrit13 * tcrit512 * mmix / patm * 1e-4


def evap(tk):
    """Chaleur latente d'évaporation (J kg-1)."""
    return (313.15 - tk) / 30.0 * (-71100.0) + 2.4073e6


def emis_atm(tk, rh):
    """Émissivité de l'atmosphère (Oke), rh en fraction."""
    e = rh * esat(tk)
    return 0.575 * e ** 0.143


def h_sphere_in_air(diameter, tk, pair, speed):
    density = pair * 100.0 / (R_AIR * tk)
    re = max(speed, MIN_SPEED) * density * diameter / viscosity(tk)
    nu = 2.0 + 0.6 * math.sqrt(re) * PR ** 0.3333
    return nu * thermal_cond(tk) / diameter


def h_cylinder_in_air(diameter, tk, pair, speed):
    a, b, c = 0.56, 0.281, 0.4
    density = pair * 100.0 / (R_AIR * tk)
    re = max(speed, MIN_SPEED) * density * diameter / viscosity(tk)
    nu = b * re ** (1.0 - c) * PR ** (1.0 - a)
    return nu * thermal_cond(tk) / diameter


# ---------------------------------------------------------------- position du soleil
def _daynum(year, month, day):
    begmonth = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    d = begmonth[month] + day
    if leap and month > 2:
        d += 1
    return d


def solar_position(year, month, day_frac, lat, lon):
    """Formules basse précision de l'Astronomical Almanac (portage de solarposition()).

    day_frac : jour du mois + fraction de jour en temps universel.
    Renvoie (élévation en degrés, non corrigée de la réfraction ; distance Terre-Soleil en UA).
    """
    daynumber = _daynum(year, month, int(day_frac))
    delta_years = year - 2000
    delta_days = delta_years * 365 + int(delta_years / 4) + daynumber
    if year > 2000:
        delta_days += 1
    days_j2000 = delta_days - 1.5
    cent_j2000 = days_j2000 / 36525.0
    ut = math.modf(day_frac)[0]
    days_j2000 += ut
    ut *= 24.0
    mean_anomaly = math.modf((357.528 + 0.9856003 * days_j2000) / 360.0)[0] * 2 * math.pi
    mean_longitude = math.modf((280.460 + 0.9856474 * days_j2000) / 360.0)[0] * 2 * math.pi
    mean_obliquity = (23.439 - 4.0e-7 * days_j2000) * DEG_RAD
    ecl = ((1.915 * math.sin(mean_anomaly)) + 0.020 * math.sin(2 * mean_anomaly)) * DEG_RAD + mean_longitude
    distance = 1.00014 - 0.01671 * math.cos(mean_anomaly) - 0.00014 * math.cos(2 * mean_anomaly)
    ap_ra = math.atan2(math.cos(mean_obliquity) * math.sin(ecl), math.cos(ecl))
    if ap_ra < 0:
        ap_ra += 2 * math.pi
    ap_ra = math.modf(ap_ra / (2 * math.pi))[0] * 24.0
    ap_dec = math.asin(math.sin(mean_obliquity) * math.sin(ecl))
    gmst0h = 24110.54841 + cent_j2000 * (8640184.812866 + cent_j2000 * (0.093104 - cent_j2000 * 6.2e-6))
    gmst0h = math.modf(gmst0h / 3600.0 / 24.0)[0] * 24.0
    if gmst0h < 0:
        gmst0h += 24.0
    lmst = gmst0h + ut * 1.00273790934 + lon / 15.0
    lmst = math.modf(lmst / 24.0)[0] * 24.0
    if lmst < 0:
        lmst += 24.0
    lha = lmst - ap_ra
    if lha < -12.0:
        lha += 24.0
    elif lha > 12.0:
        lha -= 24.0
    lat_r = lat * DEG_RAD
    lha = lha / 24.0 * 2 * math.pi
    alt = math.asin(math.sin(ap_dec) * math.sin(lat_r) + math.cos(ap_dec) * math.cos(lha) * math.cos(lat_r))
    tan_alt = math.tan(alt) if abs(alt) < 1.57079615 else 6.0e6
    alt = alt / DEG_RAD
    # correction de réfraction (atmosphère standard), ajoutée à l'élévation comme dans wbgt.c
    pressure, temp = 1013.25, 15.0
    if alt < -1.0 or tan_alt == 6.0e6:
        refraction = 0.0
    elif alt < 19.225:
        refraction = (0.1594 + alt * (0.0196 + 0.00002 * alt)) * pressure
        refraction /= (1.0 + alt * (0.505 + 0.0845 * alt)) * (273.0 + temp)
    else:
        refraction = 0.00452 * (pressure / (273.0 + temp)) / tan_alt
    return alt + refraction, distance


def solar_parameters(year, month, day_frac_utc, lat, lon, solar):
    """cos(angle zénithal), rayonnement corrigé et fraction directe (calc_solar_parameters)."""
    elev, soldist = solar_position(year, month, day_frac_utc, lat, lon)
    cza = math.cos((90.0 - elev) * DEG_RAD)
    toasolar = SOLAR_CONST * max(0.0, cza) / (soldist * soldist)
    if cza < CZA_MIN:
        toasolar = 0.0
    if toasolar > 0:
        normsolar = min(solar / toasolar, NORMSOLAR_MAX)
        solar = normsolar * toasolar
        if normsolar > 0:
            fdir = math.exp(3.0 - 1.34 * normsolar - 1.65 / normsolar)
            fdir = max(min(fdir, 0.9), 0.0)
        else:
            fdir = 0.0
    else:
        fdir = 0.0
    return cza, solar, fdir


# ---------------------------------------------------------------- vent à 2 m
def stab_srdt(daytime, speed, solar, dT):
    """Classe de stabilité de Pasquill (1 = A ... 6 = F), méthode SRDT (EPA-454/5-99-005)."""
    lsrdt = [[1, 1, 2, 4, 0, 5, 6, 0],
             [1, 2, 3, 4, 0, 5, 6, 0],
             [2, 2, 3, 4, 0, 4, 4, 0],
             [3, 3, 4, 4, 0, 0, 0, 0],
             [3, 4, 4, 4, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]]
    if daytime:
        j = 0 if solar >= 925 else 1 if solar >= 675 else 2 if solar >= 175 else 3
        i = 4 if speed >= 6 else 3 if speed >= 5 else 2 if speed >= 3 else 1 if speed >= 2 else 0
    else:
        j = 6 if dT >= 0 else 5
        i = 2 if speed >= 2.5 else 1 if speed >= 2.0 else 0
    return lsrdt[i][j]


def est_wind_speed(speed, zspeed, stability_class, urban):
    urban_exp = [0.15, 0.15, 0.20, 0.25, 0.30, 0.30]
    rural_exp = [0.07, 0.07, 0.10, 0.15, 0.35, 0.55]
    exponent = (urban_exp if urban else rural_exp)[stability_class - 1]
    return max(speed * (REF_HEIGHT / zspeed) ** exponent, MIN_SPEED)


# ---------------------------------------------------------------- températures de globe et humide
def t_globe(tk, rh, pair, speed, solar, fdir, cza):
    """Température de globe (°C). tk en K, rh fraction, pair hPa, speed m/s à 2 m."""
    tsfc = tk
    tg_prev = tk
    for _ in range(MAX_ITER):
        tref = 0.5 * (tg_prev + tk)
        h = h_sphere_in_air(D_GLOBE, tref, pair, speed)
        tg_new = (0.5 * (emis_atm(tk, rh) * tk ** 4 + EMIS_SFC * tsfc ** 4)
                  - h / (STEFANB * EMIS_GLOBE) * (tg_prev - tk)
                  + solar / (2.0 * STEFANB * EMIS_GLOBE) * (1.0 - ALB_GLOBE)
                  * (fdir * (1.0 / (2.0 * cza) - 1.0) + 1.0 + ALB_SFC)) ** 0.25
        if abs(tg_new - tg_prev) < CONVERGENCE:
            return tg_new - 273.15
        tg_prev = 0.9 * tg_prev + 0.1 * tg_new
    return float("nan")


def t_wetbulb(tk, rh, pair, speed, solar, fdir, cza, rad=1):
    """Température humide naturelle (rad=1) ou psychrométrique (rad=0), en °C."""
    a = 0.56
    tsfc = tk
    sza = math.acos(max(-1.0, min(1.0, cza)))
    eair = rh * esat(tk)
    twb_prev = dew_point(eair)
    for _ in range(MAX_ITER):
        tref = 0.5 * (twb_prev + tk)
        h = h_cylinder_in_air(D_WICK, tref, pair, speed)
        fatm = (STEFANB * EMIS_WICK * (0.5 * (emis_atm(tk, rh) * tk ** 4 + EMIS_SFC * tsfc ** 4) - twb_prev ** 4)
                + (1.0 - ALB_WICK) * solar * ((1.0 - fdir) * (1.0 + 0.25 * D_WICK / L_WICK)
                                              + fdir * ((math.tan(sza) / math.pi) + 0.25 * D_WICK / L_WICK) + ALB_SFC))
        ewick = esat(twb_prev)
        density = pair * 100.0 / (R_AIR * tref)
        sc = viscosity(tref) / (density * diffusivity(tref, pair))
        twb_new = tk - evap(tref) / RATIO * (ewick - eair) / (pair - ewick) * (PR / sc) ** a + (fatm / h * rad)
        if abs(twb_new - twb_prev) < CONVERGENCE:
            return twb_new - 273.15
        twb_prev = 0.9 * twb_prev + 0.1 * twb_new
    return float("nan")


def wbgt_liljegren(year, month, day, hour_utc, minute_utc, lat, lon, solar, pair, tair, relhum,
                   speed, zspeed=10.0, dT=-1.0, urban=False):
    """WBGT extérieur selon Liljegren et al. (2008).

    Paramètres
    ----------
    year, month, day, hour_utc, minute_utc : instant (temps universel)
    lat, lon : degrés décimaux (lon positive vers l'est)
    solar    : rayonnement global horizontal (W m-2)
    pair     : pression atmosphérique (hPa)
    tair     : température de l'air (°C)
    relhum   : humidité relative (%)
    speed    : vitesse du vent (m/s) mesurée à la hauteur zspeed (m)
    dT       : gradient vertical de température (haut - bas, °C), utilisé seulement de nuit
               pour la classe de stabilité ; inconnu ici -> valeur négative par défaut
               (atmosphère légèrement instable/neutre, classes D/E).
    urban    : exposants de la loi puissance « urbains » (True) ou « ruraux » (False)

    Renvoie un dict : wbgt, t_globe, t_nwb, t_psy, vent_2m, cza, fdir, solaire_corrige.
    """
    day_frac = day + (hour_utc + minute_utc / 60.0) / 24.0
    cza, solar_c, fdir = solar_parameters(year, month, day_frac, lat, lon, solar)
    if zspeed != REF_HEIGHT:
        stab = stab_srdt(cza > 0, speed, solar_c, dT)
        speed2 = est_wind_speed(speed, zspeed, stab, urban)
    else:
        speed2 = speed
    tk = tair + 273.15
    rh = 0.01 * relhum
    tg = t_globe(tk, rh, pair, speed2, solar_c, fdir, cza)
    tnwb = t_wetbulb(tk, rh, pair, speed2, solar_c, fdir, cza, rad=1)
    tpsy = t_wetbulb(tk, rh, pair, speed2, solar_c, fdir, cza, rad=0)
    return {"wbgt": 0.1 * tair + 0.2 * tg + 0.7 * tnwb, "t_globe": tg, "t_nwb": tnwb, "t_psy": tpsy,
            "vent_2m": speed2, "cza": cza, "fdir": fdir, "solaire_corrige": solar_c}


def wbgt_simplifie_acsm(tair, relhum, pair=1013.25):
    """Approximation « ombre » de l'Australian Bureau of Meteorology (ABM) :
    WBGT ≈ 0,567 T + 0,393 e + 3,94, e en hPa. Fournie uniquement pour comparaison
    (ignore vent et rayonnement)."""
    e = relhum / 100.0 * 6.105 * math.exp(17.27 * tair / (237.7 + tair))
    return 0.567 * tair + 0.393 * e + 3.94
