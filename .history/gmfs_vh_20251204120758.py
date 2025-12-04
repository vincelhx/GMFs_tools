import numpy as np


def gmf_rs2_v2(incidence, speed, phi=None):
    """
    Radarsat-2 VH GMF : relation between sigma0, incidence and windspeed. 

    Parameters
    ----------
    incidence: xarray.DataArray
        incidence angle [deg]
    speed: xarray.DataArray
        wind speed [m/s]

    Returns
    -------
    sigma0: xarray.DataArray 
        linear sigma0 

    """
    # constants params

    Z1_p = np.array([6.55519203e-06,  2.49753154e+00, -1.35734881e-02])
    Z2_p = np.array([1.47342197e-04, -4.07334797e-06,  3.43593382e-08,  1.10188639e+00,
                     1.40782758e-02, -1.53748743e-04])
    Final_p = np.array([-0.18675905, 24.48859492,  0.19185442, 25.38275738])

    # Z1

    a0_Z1 = Z1_p[0]
    b0_Z1 = Z1_p[1]
    b1_Z1 = Z1_p[2]

    a_Z1 = a0_Z1
    b_Z1 = b0_Z1 + b1_Z1*incidence
    sig_Z1 = a_Z1*speed**(b_Z1)

    # Z2
    a0_Z2 = Z2_p[0]
    a1_Z2 = Z2_p[1]
    a2_Z2 = Z2_p[2]
    b0_Z2 = Z2_p[3]
    b1_Z2 = Z2_p[4]
    b2_Z2 = Z2_p[5]

    a_Z2 = a0_Z2 + a1_Z2*incidence + a2_Z2*incidence**2
    b_Z2 = b0_Z2 + b1_Z2*incidence + b2_Z2*incidence**2
    sig_Z2 = a_Z2*speed**(b_Z2)

    c0, c1 = Final_p[0], Final_p[1]
    c2, c3 = Final_p[2], Final_p[3]
    sigmoid_Z1 = 1 / (1 + np.exp(-c0*(speed-c1)))
    sigmoid_Z2 = 1 / (1 + np.exp(-c2*(speed-c3)))
    sig_Final = sig_Z1 * sigmoid_Z1 + sig_Z2 * sigmoid_Z2
    return sig_Final


def gmf_s1_v2(incidence, speed, phi=None):
    """
    Sentinel-1 VH GMF : relation between sigma0, incidence and windspeed. 

    Parameters
    ----------
    incidence: xarray.DataArray
        incidence angle [deg]
    speed: xarray.DataArray
        wind speed [m/s]

    Returns
    -------
    sigma0: xarray.DataArray 
        linear sigma0 
    """

    # constants params

    Z1_p = np.array([2.13755392e-06,  2.47395267e+00, -2.85775085e-03])

    Z2_p = np.array([6.54058552e-05, -2.43845137e-06,  2.87698338e-08,  1.14509104e+00,
                     3.41828829e-02, -4.79715441e-04])
    Final_p = np.array([-0.23257086, 12.39717002,  0.21667263, 12.22862991])

    # Z1
    a0_Z1 = Z1_p[0]
    b0_Z1 = Z1_p[1]
    b1_Z1 = Z1_p[2]

    a_Z1 = a0_Z1
    b_Z1 = b0_Z1 + b1_Z1*incidence
    sig_Z1 = a_Z1*speed**(b_Z1)

    # Z2
    a0_Z2 = Z2_p[0]
    a1_Z2 = Z2_p[1]
    a2_Z2 = Z2_p[2]
    b0_Z2 = Z2_p[3]
    b1_Z2 = Z2_p[4]
    b2_Z2 = Z2_p[5]

    a_Z2 = a0_Z2 + a1_Z2*incidence + a2_Z2*incidence**2
    b_Z2 = b0_Z2 + b1_Z2*incidence + b2_Z2*incidence**2
    sig_Z2 = a_Z2*speed**(b_Z2)

    c0, c1 = Final_p[0], Final_p[1]
    c2, c3 = Final_p[2], Final_p[3]
    sigmoid_Z1 = 1 / (1 + np.exp(-c0*(speed-c1)))
    sigmoid_Z2 = 1 / (1 + np.exp(-c2*(speed-c3)))
    sig_Final = sig_Z1 * sigmoid_Z1 + sig_Z2 * sigmoid_Z2
    return sig_Final


@GmfModel.register(wspd_range=[3.0, 80.0], pol="VH", units="linear", defer=False)
def gmf_s1_v3_ew_rec(incidence, speed, phi=None):
    """
    Sentinel-1 VH GMF : relation between sigma0, incidence and windspeed.

    Parameters
    ----------
    incidence: xarray.DataArray
        incidence angle [deg]
    speed: xarray.DataArray
        wind speed [m/s]

    Returns
    -------
    sigma0: xarray.DataArray
        linear sigma0
    """

    # S1 >= 20190731 | EW TEST_1_15_smap_corrected
    u10 = speed
    inc = incidence
    a0_Z1, b0_Z1, b1_Z1 = 3.5033427638479895e-06, 2.5486758595982275, -0.009042529888607539
    a_Z1 = a0_Z1
    b_Z1 = b0_Z1 + b1_Z1 * inc
    sig_Z1 = a_Z1 * u10 ** (b_Z1)
    a0_Z2, a1_Z2, a2_Z2, b0_Z2, b1_Z2, b2_Z2 = 4.142689709809047e-05, - \
        1.6620917447744406e-06, 2.4331104610101826e-08, 1.277314996198736, 0.03813903872809897, - \
        0.0006506765114704733
    a_Z2 = a0_Z2 + a1_Z2 * inc + a2_Z2 * inc**2
    b_Z2 = b0_Z2 + b1_Z2 * inc + b2_Z2 * inc**2
    sig_Z2 = a_Z2 * u10 ** (b_Z2)
    c0, c1, c2, c3 = - \
        0.2522916645939956, 15.3393676653533, 0.24259895576004784, 15.203063214062643
    sigmoid1 = 1 / (1 + np.exp(-c0 * (u10 - c1)))
    sigmoid2 = 1 / (1 + np.exp(-c2 * (u10 - c3)))
    return 10**((10 * np.log10(sig_Z1) * sigmoid1 + 10 * np.log10(sig_Z2) * sigmoid2) / 10)


@GmfModel.register(wspd_range=[3.0, 80.0], pol="VH", units="linear", defer=False)
def gmf_rcm_v4(incidence, speed, phi=None):
    """
    Radarsat Consteallation Mission VH GMF : relation between sigma0, incidence and windspeed.
    Gmf gmf_rcm_v3 modified with a minor correection to b0_Z2.
    Parameters
    ----------
    incidence: xarray.DataArray
        incidence angle [deg]
    speed: xarray.DataArray
        wind speed [m/s]

    Returns
    -------
    sigma0: xarray.DataArray
        linear sigma0

    """

    # RCM TEST_1_20_smap_corrected
    u10 = speed
    inc = incidence
    a0_Z1, b0_Z1, b1_Z1 = 7.093964676135241e-06, 2.3722948391886542, -0.009516840375089524
    a_Z1 = a0_Z1
    b_Z1 = b0_Z1 + b1_Z1 * inc
    sig_Z1 = a_Z1 * u10 ** (b_Z1)
    a0_Z2, a1_Z2, a2_Z2, b0_Z2, b1_Z2, b2_Z2 = 6.689451099284358e-05, - \
        1.3956325894252652e-06, 9.227949977841212e-09, 1.4687699534267797, 0.005735224541037088, - \
        7.164130353316848e-05
    a_Z2 = a0_Z2 + a1_Z2 * inc + a2_Z2 * inc**2
    b_Z2 = b0_Z2*1.01 + b1_Z2 * inc + b2_Z2 * inc**2
    sig_Z2 = a_Z2 * u10 ** (b_Z2)
    c0, c1, c2, c3 = - \
        0.2454472887447197, 15.537961353644508, 0.24011368010838255, 15.332883245452303
    sigmoid1 = 1 / (1 + np.exp(-c0 * (u10 - c1)))
    sigmoid2 = 1 / (1 + np.exp(-c2 * (u10 - c3)))
    return 10**((10 * np.log10(sig_Z1) * sigmoid1 + 10 * np.log10(sig_Z2) * sigmoid2) / 10)


@GmfModel.register(wspd_range=[3.0, 80.0], pol="VH", units="linear", defer=False)
def gmf_rs2_v4(incidence, speed, phi=None):
    """
    Radarsat-2 VH GMF : relation between sigma0, incidence and windspeed.
    Gmf gmf_rs2_v3 modified with a minor correection to b0_Z2.

    Parameters
    ----------
    incidence: xarray.DataArray
        incidence angle [deg]
    speed: xarray.DataArray
        wind speed [m/s]

    Returns
    -------
    sigma0: xarray.DataArray
        linear sigma0

    """
    # RS2 TEST_1_17.5_smap_corrected
    u10 = speed
    inc = incidence
    a0_Z1, b0_Z1, b1_Z1 = 8.423384272498706e-06, 2.4351127340627374, -0.01450322326682606
    a_Z1 = a0_Z1
    b_Z1 = b0_Z1 + b1_Z1 * inc
    sig_Z1 = a_Z1 * u10 ** (b_Z1)
    a0_Z2, a1_Z2, a2_Z2, b0_Z2, b1_Z2, b2_Z2 = 0.00014955206131320428, - \
        4.737691852310481e-06, 3.813107432709729e-08, 1.524883207000445, - \
        0.01322253424944054, 0.00037527120092119504
    a_Z2 = a0_Z2 + a1_Z2 * inc + a2_Z2 * inc**2
    b_Z2 = b0_Z2*1.01 + b1_Z2 * inc + b2_Z2 * inc**2
    sig_Z2 = a_Z2 * u10 ** (b_Z2)
    c0, c1, c2, c3 = - \
        0.2222881984904166, 13.118282628673661, 0.21426139278646567, 12.768845054319682
    sigmoid1 = 1 / (1 + np.exp(-c0 * (u10 - c1)))
    sigmoid2 = 1 / (1 + np.exp(-c2 * (u10 - c3)))
    return 10**((10 * np.log10(sig_Z1) * sigmoid1 + 10 * np.log10(sig_Z2) * sigmoid2) / 10)
