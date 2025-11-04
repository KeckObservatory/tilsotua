"""
Module to remove the refraction correction from the mask results. Interpolates
the refraction correction applied by autoslit during mask creation and removes it
from the slit positions.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def refraction_calc(data,racenter,deccenter):
    """
    Remove the effect of the refraction correction from RA,Dec slit
    positions.

    Args:
        data: table
            Current table of slit data

        racenter: float
            RA of mask center in radians

        deccenter: float
            Dec of mask center in radians


    Returns:
        data: table
            Data table with updated slit positions that are equivalent
            to pre-refraction correction autoslit positions in RA and Dec

    """

    rph = np.pi*15./180 #radians per hour
    rpd = np.pi/180. #radians per degree
    dpr = 1/rpd #radians per degree
    H = 0 * rph  #chosen hour angle
    delta =  12.8/60
    ra_range = np.linspace(racenter*dpr-delta,racenter*dpr+delta, num = 300)
    dec_range = np.linspace(deccenter*dpr-delta,deccenter*dpr+delta, num = 300)
    ra_grid = np.zeros(shape=(len(ra_range),len(dec_range)))
    dec_grid = np.zeros(shape=(len(ra_range),len(dec_range)))
    for i in range(len(ra_grid)):
        for j in range(len(dec_grid)):
            ra_grid[i:j]=ra_range[i]
            dec_grid[i:j]=dec_range[j]
    ra_grid,dec_grid = np.meshgrid(ra_range, dec_range, indexing='ij',sparse=True)

    #calculate correction values
    HA = H + racenter-ra_range*rpd

    #======================================================================================
    def ra_rad_refraction(ra,dec,HA):
        """
        Function to calculate the refraction correction in the RA position for a range of mask positions

        Args:
            ra: array
                1D array of RA positions on mask

            dec: array
                1D array of Dec positions on mask

            HA: float
                Hour angle selected for mask

        Returns:
            1D Array of the refraction correction values in degrees for the RA position.
        """
        rpd = np.pi/180. #radians per degree
        dpr = 1/rpd #degrees per radian
        rpas = rpd/3600. #radians per arcsec
        pres = 486.  #atmos pressure
        wave = 6000./10000  #wavelength
        lat = 19.828 * rpd #keck latitude
        temp = 0
        ra = ra*rpd
        dec = dec*rpd
        #calculate alt and zenith distance

        sina = np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(HA)
        cosz = sina
        sinz = np.sqrt(1.-sina**2)
        tanz = sinz/cosz


        #calculate parallactic angle
        sinq = np.cos(lat)*np.sin(HA)/sinz

        #calculate refractive index
        wavers = 1/(wave**2)
        N = 64.328 + (29498.1/(146.-wavers)) + (255.4/(41.-wavers))
        N = (1*10**-6)*N

        #calculate temp and pressure correction
        tcorr = 1+0.003661*temp
        num = 720.88*tcorr
        N = N * ((pres * (1.+((1.049-0.0157*temp) * 10**-6 * pres))) / num)#N*((pres*(1.+(1.049-((0.0157*temp)*(1*10**-6)*pres))))/num)

        #calculate refraction
        R = N * 206265 * tanz

        #calculate correction to RA and Dec
        DA = R * sinq * rpas/np.cos(dec)
        return(DA*dpr)

    #======================================================================================

    def ra_rad_refraction_grid(ra,dec,HA):
        """
        Function to calculate the refraction correction in the RA position for a range of mask positions

        Args:
            ra: array
                1D array of RA positions on mask

            dec: array
                1D array of Dec positions on mask

            HA: float
                Hour angle selected for mask

        Returns:
            2D Array of the refraction correction values in degrees for the RA position.
        """
        rpd = np.pi/180. #radians per degree
        dpr = 1/rpd #degrees per radian
        rpas = rpd/3600. #radians per arcsec
        pres = 486.  #atmos pressure
        wave = 6000./10000  #wavelength
        lat = 19.828 * rpd #keck latitude
        temp = 0
        ra = ra*rpd
        dec = np.swapaxes(dec*rpd,0,1)
        corrections = np.zeros(shape=(len(ra),len(ra)))

        #calculate alt and zenith distance
        for i in range(len(ra)-1):
            for j in range(len(ra)-1):
                sina = np.sin(lat)*np.sin(dec[i])+np.cos(lat)*np.cos(dec[i])*np.cos(HA[j])
                cosz = sina
                sinz = np.sqrt(1.-sina**2)
                tanz = sinz/cosz


                #calculate parallactic angle
                sinq = np.cos(lat)*np.sin(HA[j])/sinz

                #calculate refractive index
                wavers = 1/(wave**2)
                N = 64.328 + (29498.1/(146.-wavers)) + (255.4/(41.-wavers))
                N = (1*10**-6)*N


                #calculate temp and pressure correction
                tcorr = 1+0.003661*temp
                num = 720.88*tcorr
                N = N * ((pres * (1.+((1.049-0.0157*temp) * 10**-6 * pres))) / num)#N*((pres*(1.+(1.049-((0.0157*temp)*(1*10**-6)*pres))))/num)

                #calculate refraction
                R = N * 206265 * tanz

                #calculate correction to RA and Dec
                DA = R * sinq * rpas/np.cos(dec[i])

                corrections[i,j] = DA*dpr

        return(np.swapaxes(corrections,0,1))

    #======================================================================================
    def dec_rad_refraction(ra,dec,HA):
        """
        Function to calculate the refraction correction in the Dec position for a range of mask positions

        Args:
            ra: array
                1D array of RA positions on mask

            dec: array
                1D array of Dec positions on mask

            HA: float
                Hour angle selected for mask

        Returns:
            1D Array of the refraction correction values in degrees for the Dec position.
        """
        rpd = np.pi/180. #radians per degree
        dpr = 1/rpd #degrees per radian
        rpas = rpd/3600. #radians per arcsec
        pres = 486.  #atmos pressure
        wave = 6000./10000  #wavelength
        lat = 19.828 * rpd #keck latitude
        temp = 0
        dec = dec*rpd
        #calculate alt and zenith distance
        sina = np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(HA)
        cosz = sina
        sinz = np.sqrt(1.-sina**2)
        tanz = sinz/cosz


        #calculate parallactic angle
        #if sinz != 0:
        sinq = np.cos(lat)*np.sin(HA)/sinz
        cosq = (np.cos(np.pi/2-lat) - cosz *np.cos(np.pi/2-dec))/(sinz*np.sin(np.pi/2-dec))

        #calculate refractive index
        wavers = 1/(wave**2)
        N = 64.328 + (29498.1/(146.-wavers)) + (255.4/(41.-wavers))
        N = (1*10**-6)*N


        #calculate temp and pressure correction
        tcorr = 1+0.003661*temp
        num = 720.88*tcorr
        N = N * ((pres * (1.+((1.049-0.0157*temp) * 10**-6 * pres))) / num)#N*((pres*(1.+(1.049-((0.0157*temp)*(1*10**-6)*pres))))/num)

        #calculate refraction
        R = N * 206265 * tanz

        #calculate correction to RA and Dec
        DD = R * cosq * rpas
        return(DD*dpr)

    def dec_rad_refraction_grid(ra,dec,HA):
        """
        Function to calculate the refraction correction in the Dec position for a range of mask positions

        Args:
            ra: array
                1D array of RA positions on mask

            dec: array
                1D array of Dec positions on mask

            HA: float
                Hour angle selected for mask

        Returns:
            2D Array of the refraction correction values in degrees for the Dec position.
        """
        rpd = np.pi/180. #radians per degree
        dpr = 1/rpd #degrees per radian
        rpas = rpd/3600. #radians per arcsec
        pres = 486.  #atmos pressure
        wave = 6000./10000  #wavelength
        lat = 19.828 * rpd #keck latitude
        temp = 0
        ra = ra*rpd
        dec = np.swapaxes(dec*rpd,0,1)
        corrections = np.zeros(shape=(len(ra),len(ra)))
        #calculate alt and zenith distance
        for i in range(len(dec)-1):
            for j in range(len(ra)-1):
                sina = np.sin(lat)*np.sin(dec[i])+np.cos(lat)*np.cos(dec[i])*np.cos(HA[j])
                cosz = sina
                sinz = np.sqrt(1.-sina**2)
                tanz = sinz/cosz


                #calculate parallactic angle
                cosq = (np.cos(np.pi/2-lat) - cosz *np.cos(np.pi/2-dec[i]))/(sinz*np.sin(np.pi/2-dec[i]))

                #calculate refractive index
                wavers = 1/(wave**2)
                N = 64.328 + (29498.1/(146.-wavers)) + (255.4/(41.-wavers))
                N = (1*10**-6)*N


                #calculate temp and pressure correction
                tcorr = 1+0.003661*temp
                num = 720.88*tcorr
                N = N * ((pres * (1.+((1.049-0.0157*temp) * 10**-6 * pres))) / num)#N*((pres*(1.+(1.049-((0.0157*temp)*(1*10**-6)*pres))))/num)

                #calculate refraction
                R = N * 206265 * tanz

                #calculate correction to RA and Dec
                DD = R * cosq * rpas
                corrections[i,j] = DD*dpr

        return(np.swapaxes(corrections,0,1))

    #======================================================================================
    #Apply the correction to the slit positions in the data array
    #Set up the correction values on a grid of already corrected values for comparison
    corr_ra_range = ra_range+ra_rad_refraction(ra_range,dec_range,HA)
    corr_dec_range = dec_range+dec_rad_refraction(ra_range,dec_range,HA)
    ref_data = ra_rad_refraction_grid(ra_grid, dec_grid,HA)
    dec_ref_data = dec_rad_refraction_grid(ra_grid,dec_grid,HA)

    #call the actual interpolation function
    interp = RegularGridInterpolator((corr_ra_range, corr_dec_range), ref_data,method='linear',bounds_error=True)
    decinterp = RegularGridInterpolator((corr_ra_range,corr_dec_range), dec_ref_data,method='linear',bounds_error=True)
    for i in range(len(data['Calc_RA'])):
        data['Calc_RA'][i] -= interp([data['Calc_RA'][i],data['Calc_Dec'][i]])
        data['Calc_Dec'][i] -= decinterp([data['Calc_RA'][i],data['Calc_Dec'][i]])

    return(data)
