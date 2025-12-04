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
    #constants params 
    
    Z1_p = np.array([ 6.55519203e-06,  2.49753154e+00, -1.35734881e-02])
    Z2_p =  np.array([ 1.47342197e-04, -4.07334797e-06,  3.43593382e-08,  1.10188639e+00,
        1.40782758e-02, -1.53748743e-04])
    Final_p =  np.array([-0.18675905, 24.48859492,  0.19185442, 25.38275738])

    #Z1

    a0_Z1 = Z1_p[0]
    b0_Z1 = Z1_p[1]
    b1_Z1 = Z1_p[2]
    
    a_Z1 = a0_Z1 
    b_Z1 = b0_Z1 + b1_Z1*incidence 
    sig_Z1 = a_Z1*speed**(b_Z1) 
    
    #Z2
    a0_Z2 = Z2_p[0]
    a1_Z2 = Z2_p[1]
    a2_Z2 = Z2_p[2]
    b0_Z2 = Z2_p[3]
    b1_Z2 = Z2_p[4]
    b2_Z2 = Z2_p[5]
    
    a_Z2 = a0_Z2 + a1_Z2*incidence + a2_Z2*incidence**2 
    b_Z2 = b0_Z2 + b1_Z2*incidence + b2_Z2*incidence**2
    sig_Z2 = a_Z2*speed**(b_Z2)

    c0,c1 = Final_p[0],Final_p[1]
    c2,c3 = Final_p[2],Final_p[3]
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
    
    #constants params 
    
    Z1_p = np.array([ 2.13755392e-06,  2.47395267e+00, -2.85775085e-03])

    Z2_p =  np.array([ 6.54058552e-05, -2.43845137e-06,  2.87698338e-08,  1.14509104e+00,
        3.41828829e-02, -4.79715441e-04])
    Final_p = np.array([-0.23257086, 12.39717002,  0.21667263, 12.22862991])

    
    #Z1
    a0_Z1 = Z1_p[0]
    b0_Z1 = Z1_p[1]
    b1_Z1 = Z1_p[2]
    
    a_Z1 = a0_Z1 
    b_Z1 = b0_Z1 + b1_Z1*incidence 
    sig_Z1 = a_Z1*speed**(b_Z1) 
    
    #Z2
    a0_Z2 = Z2_p[0]
    a1_Z2 = Z2_p[1]
    a2_Z2 = Z2_p[2]
    b0_Z2 = Z2_p[3]
    b1_Z2 = Z2_p[4]
    b2_Z2 = Z2_p[5]
    
    a_Z2 = a0_Z2 + a1_Z2*incidence + a2_Z2*incidence**2 
    b_Z2 = b0_Z2 + b1_Z2*incidence + b2_Z2*incidence**2
    sig_Z2 = a_Z2*speed**(b_Z2)

    c0,c1 = Final_p[0],Final_p[1]
    c2,c3 = Final_p[2],Final_p[3]
    sigmoid_Z1 = 1 / (1 + np.exp(-c0*(speed-c1)))
    sigmoid_Z2 = 1 / (1 + np.exp(-c2*(speed-c3)))
    sig_Final = sig_Z1 * sigmoid_Z1 + sig_Z2 * sigmoid_Z2 
    return sig_Final
