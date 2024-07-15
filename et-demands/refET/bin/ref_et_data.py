"""ref_et_data.py
Defines refET class
Called by met_nodes.py

"""

import math
import numpy as np
import refet

class refET:

    def __init__(self, b0, b1, b2):
        """ """

        # Thorton and Running Coefficients

        self.TR_b0 = b0
        self.TR_b1 = b1
        self.TR_b2 = b2

        # these were defined as 'static' variable within a Function in the VB.net version

        self._Tp3 = 0.0 # static
        self._Tp2 = 0.0 # static
        self._Tp1 = 0.0 # static
        self._ndays = 0 # static
        self._lastDay = 0 # static

    def _latent_heat_vaporization(self, tmean):
        """Latent heat of vaporization

        Parameters
        ----------
        tmean : scalar or array_like of shape(M, )
            mean daily temperature [C]

        Returns
        -------
            lmbda : ndarray
                latent heat of evaporation [J kg-1]

        Notes
        -----
        In ASCE-NWRI (2005):
            A constant value of lmbda = 2.45 MJ kg-1 is used

        """

        lmbda = 2500000.0 - 2360 * tmean  # J/Kg   Latent Heat of Vaporization
        return lmbda

    def _psychro_const(self, pair, lmbda):
        """Psychrometric constant (Eq.35 mod.)

        Parameters
        ----------
        pair : scalar or array_like of shape(M, )
            Air pressure [kPa].
        lambda : scalar or array_like of shape(M, )
            latent heat of evaporation [J kg-1]

        Returns
        -------
        psy : ndarray
            Psychrometric constant [kPa C-1].

        Notes
        -----
        In ASCE-NWRI (2005):
            This is modified from Eq. 35 as lmbda is not fixed

        """

        psy = 1013 * pair / 0.622 / lmbda
        return psy

    def et_hargreaves_samani(self, doy, tmax, tmin, latitude):
        """Hargreaves Samani reference ET (https://doi.org/10.13031/2013.26773)

        Parameters
        ----------
        doy : scalar or array_like of shape(M, )
            Day of year.
        tmax : scalar or array_like of shape(M, )
            Daily maximum air temperature [C].
        tmin : scalar or array_like of shape(M, )
            Daily minimum air temperature [C].
        latitude : scalar or array_like of shape(M, )
            Latitude [deg].

        Returns
        -------
            EToHargSam : ndarray
            Hargreaves Samani grass reference ET [mm d-1]

        """

        tmean = 0.5 * (tmax + tmin)

        # compute latent heat of vaporization
        lmbda = self._latent_heat_vaporization(tmean)

        # Latutude in radians
        latRad = latitude * math.pi / 180.0  # Lat is station latitude in degrees

        # Extraterrestrial radiation
        ra = refet.calcs._ra_daily(latRad, doy, method='refet')

        if tmax < tmin:
            EToHargSam = 0.0
        else:
            EToHargSam = 0.0023 * (tmax - tmin)**0.5 * ra * (tmean + 17.8) # same units as Ra
        EToHargSam = EToHargSam / lmbda * 1000000.0 # mm/day
        return EToHargSam

    def _et_penman(self, pair, ea, lmbda, delta, tmean, Rn56, G56, u242):
        """Penman 1948 reference ET (https://doi.org/10.1098/rspa.1948.0037)

        Parameters
        ----------
        pair : scalar or array_like of shape(M, )
            Air pressure [kPa].
        ea : scalar or array_like of shape(M, )
            Actual vapor pressure [kPa].
        lmbda : scalar or array_like of shape(M, )
            latent heat of evaporation [J kg-1].
        delta : scalar or array_like of shape(M, )
            Slope of the saturation vapor pressure-temperature curve [kPa C-1].
        tmean : scalar or array_like of shape(M, )
            mean daily temperature [C].
        Rn56 : scalar or array_like of shape(M, )
            Net radiation from FAO56 [MJ m-2 d-1].
        G56 : scalar or array_like of shape(M, )
            Soil heat flux from FAO56 [MJ m-2 d-1].
        u242 : scalar or array_like of shape(M, )
            24 hr wind speed at 2 m height over grass [m s-1].

        Returns
        -------
        EToPen : ndarray
            Penman 1948 grass reference ET

        """

        # Saturation vapor pressure from tmean
        es = refet.calcs._sat_vapor_pressure(tmean) # different from other eqns. check this
        gamma56 = 0.000665 * pair # kPa/C   May 17 1999

        # Penman 1948 withorig wind Rome wind function added 8/17/09 by Justin Huntington

        # Penman = ((delta / (delta + gamma56) * (Rn56 - G56)) + 6.43 * gamma56 / (delta + gamma56) * (1 + 0.537 * U242) * (ea - ea))
        # EToPen = ((delta / (delta + gamma56)) * (Rn56 - G56) + 6.43 * (gamma56 / (delta + gamma56)) * (1 + 0.537 * u242) * (es - ea)) / lmbda


        EToPen = (((delta / (delta + gamma56) * (Rn56 - G56))) / lmbda * 1000000.0) + ((gamma56 / (delta + gamma56) * (0.26 * (1 + 0.537 * u242)) * ((es * 10) - (ea * 10))))
        EToPen = max(EToPen, 0.0)
        return(EToPen)

    def _et_fao50(self, pair, ea, es, delta, tmean, G56, Rn56, u242):
        """FAO56 Penman Monteith reference ET

        Parameters
        ----------
        pair : scalar or array_like of shape(M, )
            Air pressure [kPa].
        ea : scalar or array_like of shape(M, )
            Actual vapor pressure [kPa].
        es : scalar or array_like of shape(M, )
            Saturation vapor pressure [kPa].
        delta: scalar or array_like of shape(M, )
            Slope of the saturation vapor pressure-temperature curve [kPa C-1].
        tmean : scalar or array_like of shape(M, )
            mean daily temperature [C].
        G56 : scalar or array_like of shape(M, )
            Soil heat flux from FAO56 [MJ m-2 d-1].
        Rn56 : scalar or array_like of shape(M, )
            Net solar radiation from FAO56 [MJ m-2 d-1].
        u242 : scalar or array_like of shape(M, )
            24 hr wind speed at 2 m height over grass [m s-1].

        Returns
        -------
        EToFAO56 : ndarray
            FAO56 Penman Monteith grass reference ET

        """

        gamma56 = 0.000665 * pair # kPa/C   May 17 1999
        EToFAO56 = (0.408 * delta * (Rn56 - G56) + gamma56 * 900 / (tmean + 273) * u242 * (es - ea)) / (delta + gamma56 * (1 + 0.34 * u242)) #  may 17 1999 move before ea=fnes(TAvg)
        EToFAO56 = max(EToFAO56, 0.0)
        return(EToFAO56)

    def _et_kimberly_penman(self, lat, doy, ea, es, lmbda, delta, gamma, G82, G82o, Rn82, u242):
        """Kimberly Penman 1982 reference ET (https://eprints.nwisrl.ars.usda.gov/id/eprint/382)

        Parameters
        ----------
        lat : scalar or array_like of shape(M, )
            Latitude [deg].
        doy : scalar or array_like of shape(M, )
            Day of year.
        ea : scalar or array_like of shape(M, )
            Actual vapor pressure [kPa].
        es : scalar or array_like of shape(M, )
            Saturation vapor pressure [kPa].
        lmbda: scalar or array_like of shape(M, )
            latent heat of evaporation [J kg-1].
        delta: scalar or array_like of shape(M, )
            Slope of the saturation vapor pressure-temperature curve [kPa C-1].
        gamma : scalar or array_like of shape(M, )
            Psychrometric constant [kPa C-1].
        G82 : scalar or array_like of shape(M, )
            Soil heat flux from Wright (1982) [MJ m-2 d-1].
        G82o : scalar or array_like of shape(M, )
            Soil heat flux from Wright (1982) [MJ m-2 d-1].
        Rn82 : scalar or array_like of shape(M, )
            Net solar radiation from Wright (1982) [MJ m-2 d-1].
        u242 : scalar or array_like of shape(M, )
            Elevation [m].

        Returns
        -------
        ETrKimbPen : ndarray
            Kimberly Penman grass reference ET [mm d-1].
        EToKimbPen : ndarray
            Kimberly Penman alfalfa reference ET [mm d-1].

        Notes
        -----

        """

        j = doy
        if lat < 0:
            j = doy - 182
            if j < 1:
                j = j + 365
        Awk = 0.4 + 1.4 * math.exp(-((j - 173) / 58) ** 2)
        Bwk = (0.007 + 0.004 * math.exp(-((j - 243) / 80) ** 2)) * 86.4 # for m/s

        ETrKimbPen = (delta * (Rn82 - G82) + gamma * 6.43 * (es - ea) * (Awk + Bwk * u242)) / (delta + gamma)
        ETrKimbPen = ETrKimbPen / lmbda * 1000000.0
        ETrKimbPen = max(ETrKimbPen, 0.0)

        awkg = 0.3 + 0.58 *  math.exp(-((j - 170) / 45) ** 2) # Wright(1996) for grass
        bwkg = 0.32 + 0.54 * math.exp(-((j - 228) / 67) ** 2) # for m/s
        EToKimbPen = (delta * (Rn82 - G82o) + gamma * 6.43 * (es - ea) * (awkg + bwkg * u242)) / (delta + gamma)
        EToKimbPen = EToKimbPen / lmbda * 1000000.0
        EToKimbPen = max(EToKimbPen, 0.0)

        return (EToKimbPen, ETrKimbPen)

    def _et_priestly_taylor(self, pair, lmbda, delta, G56, Rn56):
        """Priestley Taylor grass reference ET (https://doi.org/10.1175/1520-0493(1972)100%3C0081:OTAOSH%3E2.3.CO;2)

        Parameters
        ----------
        pair : scalar or array_like of shape(M, )
            Air pressure [kPa].
        lmbda : ndarray
            latent heat of evaporation [J kg-1]
        delta : ndarray
            Slope [kPa C-1].
        G56 : scalar or array_like of shape(M, )
            Soil heat flux from FAO56.
        Rn56 : scalar or array_like of shape(M, )
            Net solar radiation from FAO56 [MJ m-2 d-1].

        Returns
        -------
        EToPriTay : ndarray
            Priestley-Taylor grass reference ET [mm d-1]

        Notes
        -----
        FAO56 refs for G, Rn, gamma

        """

        # Gamma calculated from FAO56
        gamma56 = 0.000665 * pair # kPa/C   May 17 1999
        EToPriTay = 1.26 * (delta / (delta + gamma56) * (Rn56 - G56)) / lmbda * 1000000.0
        EToPriTay = max(EToPriTay, 0.0)
        return(EToPriTay)

    def compute_penmans(self, yr, mo, da, doy, time_step, tmax, tmin, tdew, rs, u24, elev, latitude):
        """ Compute reference ET by Penman methods

        Parameters
        ----------
        yr : scalar or array_like of shape(M, )
            year
        mo : scalar or array_like of shape(M, )
            month
        da : scalar or array_like of shape(M, )
            day
        doy : scalar or array_like of shape(M, )
            day of year
        time_step : scalar or array_like of shape(M, )
            time step
        tmax : scalar or array_like of shape(M, )
            Daily maximum air temperature [C].
        tmin : scalar or array_like of shape(M, )
            Daily minimum air temperature [C].
        rs : scalar or array_like of shape(M, )
            Incoming solar radiation [MJ m-2 d-1].
        u24 : scalar or array_like of shape(M, )
            24-hour mean windspeed [m s-1]
        elev : scalar or array_like of shape(M, )
            elevation [m]
        latitude : scalar or array_like of shape(M, )
            latitude [deg]

        Returns
        -------
        Penman
            1948 Penman grass reference ET [mm d-1]
        PreTay : ndarray
            Priestley-Taylor grass reference ET [mm d-1]
        KimbPeng : ndarray
            Kimberly Penman grass reference ET [mm d-1]
        ASCEr : ndarray
            ASCE-EWRI standardized alfalfa reference ET [mm d-1]
        ASCEo : ndarray
            ASCE-EWRI standardized grass reference ET [mm d-1]
        FAO56PM : ndarray
            FAO56 Penman Monteith reference ET [mm d-1]
        KimbPen : ndarray
            Kimberly Penman alfalfa reference ET [mm d-1]

        """

        # Mean temperature
        tmean = 0.5 * (tmax + tmin)

        # Latent heat of vaporization
        lmbda = self._latent_heat_vaporization(tmean)

        # Latutude in radians
        latRad = latitude * math.pi / 180.0  # Lat is station latitude in degrees

        # Extraterrestial radiation
        ra = refet.calcs._ra_daily(latRad, doy, method="refet")

        # Air pressure from elevation
        pair = refet.calcs._air_pressure(elev, method="refet")

        # Slope of saturation pressure temperature curve
        delta = refet.calcs._es_slope(tmean, method='refet') # kPa/C

        # Actual vapor pressure approximated from dewpoint temperature
        ea = refet.calcs._sat_vapor_pressure(tdew)

        # Mean saturation vapor pressure
        es = (refet.calcs._sat_vapor_pressure(tmax) + refet.calcs._sat_vapor_pressure(tmin)) * 0.5

        # Psychrometric constant
        gamma = self._psychro_const(pair, lmbda)

        # Clear sky radiation
        rso = refet.calcs._rso_daily(ra, es, pair, doy, latRad)

        # Net radiation from FAO56 and Kimberly (1982)
        Rn56, Rn82 = self._rn_daily(elev, latitude, doy, ra, rs, rso, tmax, tmin, es)

        # Soil heat flux
        G56, G56r, G82, G82o = self._soil_heat(da, time_step, tmean, Rn56, Rn82) # modified 6/24/99

        # Wind speed adjustments for different vegetation types
        # If Ref <= 0.5 Then ht = GrassHt Else ht = AlfHt
        AlfHt = 0.5 # m
        GrassHt = 0.12 # m
        htr = AlfHt
        ht = GrassHt # assume default is grass

        # Assuming weather measurement surface is that of reference
        site_hto = ht
        site_htr = htr

        # Wind measurement height
        zw = 2 # m

        # Theorectical adjustment (Allen) over grass is:
        # d = .67 * GrassHt  'site_ht 'modified 12/21/99 to reflect wind required over grass in std eqns.
        # zom = .123 * GrassHt  'site_ht

        # U242 = U24 * Log((2 - d) / zom) / Log((zw - 0.67 * site_hto) / (0.123 * site_hto))
        # for Consuse comp, all wind is at 2 m:

        u242 = u24
        u242r = u24

        # 1948 Penman
        Penman = self._et_penman(pair, ea, lmbda, delta, tmean, Rn56, G56, u242)

        # FAO56 Penman Monteith
        FAO56PM = self._et_fao50(pair, ea, es, delta, tmean, G56, Rn56, u242)

        # 1982 Kimberly Penman
        KimbPeng, KimbPen = self._et_kimberly_penman(latitude, doy, ea, es, lmbda, delta, gamma, G82, G82o, Rn82, u242)

        # Priestley Taylor
        PreTay = self._et_priestly_taylor(pair, lmbda, delta, G56, Rn56)

        # ASCE-EWRI Penman Monteith
        ASCEPMstdr = refet.Daily(tmin, tmax, ea, rs, u242, zw, elev, latitude, doy, method='refet').etr()
            # input_units={'tmin': 'C', 'tmax': 'C', 'rs': 'mj m-2 d-1', 'uz': 'm s-1',
            #              'lat': 'deg'}
        ASCEPMstdo = refet.Daily(tmin, tmax, ea, rs, u242, zw, elev, latitude, doy, method='refet').eto()
        return (Penman, PreTay, KimbPeng, ASCEPMstdr, ASCEPMstdo, FAO56PM, KimbPen)

    def _rn_daily(self, elev, lat, doy, ra, rs, rso, tmax, tmin, es):
        """FAO56 and Kimberly 1982 net radiation

        Parameters
        ----------
        elev : scalar or array_like of shape(M, )
            elevation [m].
        latitude : scalar or array_like of shape(M, )
            latitude [deg].
        doy : scalar or array_like of shape(M, )
            day of year.
        ra : scalar or array_like of shape(M, )
            extraterrestrial radiation [MJ m-2 d-1].
        rs : scalar or array_like of shape(M, )
            measured solar radiation [MJ m-2 d-1].
        rso : scalar or array_like of shape(M, )
            clear sky radiation [MJ m-2 d-1].
        tmax : scalar or array_like of shape(M, )
            Daily maximum air temperature [C].
        tmin : scalar or array_like of shape(M, )
            Daily minimum air temperature [C].
        es : scalar or array_like of shape(M, )
            Saturation vapor pressure [kPa].

        Returns
        -------
        Rn56 : ndarray
            FA056 net radiation [MJ m-2 d-1].
        Rn82 : ndarray
            Rn82 net radiation [MJ m-2 d-1].

        """

        j = doy
        if lat < 0:
            j = doy - 182
            if j < 1:
                j = j + 365
        Rna1 = 0.26 + 0.1 * math.exp(-(0.0154 * (j - 180)) ** 2)
        Rso75 = 0.75 * ra
        Rso56 = (0.75 + 0.00002 * elev) * ra # 4/6/94
        if rso > 0:
            RsRso = rs / rso # 24-hours
        else:
            RsRso = 0.7
        if Rso56:
            RsRso56 = rs / Rso56 # 24-hours
        else:
            RsRso56 = 0.7
        RsRso = max(0.2, min(RsRso, 1.0))
        RsRso56 = max(0.2, min(RsRso56, 1.0))
        RsRso2use = RsRso # useRso based on sun angle and water vapor as of 9/25/2000
        if RsRso2use > 0.7:
            Rna = 1.126
            Rnb = -0.07
        else:
            Rna = 1.017
            Rnb = -0.06
        Rbo = 0.000000004903 * 0.5 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) * (Rna1 - 0.139 * math.sqrt(es))
        Rb = (Rna * RsRso2use + Rnb) * Rbo
        alpha = 0.29 + 0.06 * math.sin((j + 96) / 57.3)
        Rn82 = (1 - alpha) * rs - Rb

        # FAO 56 Net Radiation computation

        Rbo56 = 0.000000004903 * 0.5 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) * (0.34 - 0.14 * math.sqrt(es))
        Rnl56 = (1.35 * RsRso2use - 0.35) * Rbo56
        Rn56 = (1 - 0.23) * rs - Rnl56
        return Rn56, Rn82

    def _soil_heat(self, da, time_step, tmean, Rn56, Rn82): # modified 6/24/99
        """FAO56 and Kimberly 1982 soil heat flux

        Parameters
        ----------
        da : scalar or array_like of shape(M, )
            day
        time_step : scalar or array_like of shape(M, )
            time step
        tmean : scalar or array_like of shape(M, )
            mean daily temperature [C].
        Rn56 : scalar or array_like of shape(M, )
            FAO56 net radiation [MJ m-2 d-1].
        Rn82 : scalar or array_like of shape(M, )
            Kimberly 1892 net radiation [MJ m-2 d-1].

        Returns
        -------
        G56 : scalar or array_like of shape(M, )
            FAO56 soil heat flux [MJ m-2 d-1].
        G56r : scalar or array_like of shape(M, )
            FAO56 soil heat flux [MJ m-2 d-1].
        G82 : scalar or array_like of shape(M, )
            Kimberly 1982 soil heat flux [MJ m-2 d-1].
        G82o : scalar or array_like of shape(M, )
            Kimberly 1982 soil heat flux [MJ m-2 d-1].

        """

        # ByRef G82 As Double, ByRef G82o As Double, ByRef G56 As Double, ByRef G56r As Double
        # Static Tp3, Tp1, Tp2 As Double
        # Static ndays, lastDay As Long
        # --> 'Static' means value is persistant between calls, ie, retains its value
        # use class variable to make static
        if time_step == 'day':   # Daily or Monthly Time Step  ' added 10/13/90  start if1
            if self._lastDay < 1:  # start if2
                G82 = 0.0
                G82o = 0.0
                G56 = 0.0
                G56r = 0.0
                self._Tp3 = 0.0
                self._Tp2 = 0.0
                self._Tp1 = 0.0
                self._ndays = 0
            else:
                if (self._lastDay == 15 and da == 15) or (self._lastDay == 0 and da == 0):
                    # monthly heat flux

                    G82 = 0.0 # assume monthly heat flux is zero
                    G82 = 0.14 * (TAvg - self._Tp1) # added 3/28/94  (use prior month) (FAO)
                    G56 = G82
                    G82o = G82
                    G56r = G82
                else:
                    if self._ndays > 0:
                        G82 = (tmean - (self._Tp3 + self._Tp2 + self._Tp1) / self._ndays) * 0.3768 # MJ/m2/d
                    else:
                        G82 = 0.0
                    G56 = 0.0
                    G82o = G82
                    G56r = G56
                #  Tp3 = Tp2   ' moved after end of this end if 3/28/94
                #  Tp2 = Tp1
                #  Tp1 = tmean
                #  ndays = ndays + 1
                #  IF ndays > 3 THEN ndays = 3
            self._lastDay = da
            self._Tp3 = self._Tp2 # moved here 3/28/94
            self._Tp2 = self._Tp1
            self._Tp1 = tmean
            self._ndays = self._ndays + 1
            self._ndays = min(self._ndays, 3)
        else: # hourly time step (assumed)
            # changed Rn to Rn82 dlk 01/27/2016

            if Rn82 >= 0:  # added 10/13/90  (units are MJ/m2/day, but will be W/m2 on console.writeout)
                # If Ref < 0.5 Then
                G82o = 0.1 * Rn82 # Clothier (1986), Kustas et al (1989) in Asrar Remote Sens. Txt

                # for full cover alfalfa (but high for alfalfa. use for grass

                # Else
                G82 = 0.04 * Rn82 # for alfalfa reference based on Handbook Hydrol.
                # End If
            else:
                # If Ref < 0.5 Then
                G82o = Rn82 * 0.5 # R.Allen obersvations for USU Drainage Farm.  Alta Fescue

                # crop near full cover.  Adj. Soil heat flux assuming that
                # therm. cond. in top 100 mm was 2 MJ/m2/C.

                # Else
                G82 = Rn82 * 0.2 # alfalfa
                # End If
                # G56 = G
            if Rn56 >= 0:      # added 10/13/90  (units are MJ/m2/day, but will be W/m2 on console.writeout)
                # If Ref < 0.5 Then
                G56 = 0.1 * Rn56 # Clothier (1986), Kustas et al (1989) in Asrar Remote Sens. Txt

                # for full cover alfalfa (fits grass better than alfalfa.  use for grass)

                # Else
                G56r = 0.04 * Rn56 # alfalfa reference based on Handbook Hydrol.
                # End If
            else:
                # If Ref < 0.5 Then
                G56 = Rn56 * 0.5 # R.Allen obersvations for USU Drainage Farm.  Alta Fescue

                # crop near full cover.  Adj. Soil heat flux assuming that
                # therm. cond. in top 100 mm was 2 MJ/m2/C.

                # Else
                G56r = Rn56 * 0.2 # alfalfa
                # End If
        return G56, G56r, G82, G82o

##############
# removed tests 10/2018
