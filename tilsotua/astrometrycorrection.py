"""
Module to calculate, then remove the distortion correction applied at each slit
position on the mask. This is done with linear interpolation.
"""

import numpy as np
from astropy.table import Table,Column
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
import matplotlib
from astropy.io import ascii,fits
def astrometry_calc(data,ra_center,dec_center):
    """
    Calculates the distortion correction applied at each mask location and subtracts
    the correction from the current slit positions.

    Args:
        data: table
            Mask information table

        ra_center: float
            RA of mask center

        dec_center: float
            Dec of mask center

    Returns:
        data: table
            Mask information table with distortion correction removed from slit
                      positions.
    """
    #create a grid of values to create the points for which the correction will be calculated
    x_range = np.linspace(173,437, num = 300)#np.linspace(174,436, num = 100)
    y_range = np.linspace(-173,175, num = 300)#np.linspace(-174,174, num = 100)
    x_grid,y_grid = np.meshgrid(x_range, y_range, indexing='ij',sparse=True)
    #set up arrays to hold results later
    x_values = []
    y_values = []
    x_offsets = []
    y_offsets = []
    offsets = []
#=================================================================================================================================
    #create the routine for the astrometry correction
    def x_astrometry_correction_calc(XIN,YIN):
        """
        Function to calculate the distortion correction in the X position for a range of mask positions

        Args:
            XIN: array
                1D array of X positions on mask

            YIN: array
                1D array of Y positions on mask

        Returns:
            Array of the distortion correction values for the X position.
        """
        #set some constants
        CCD_SIDE = 2048.0
        CCD_SCALE = 0.15767
        X_CENTER = 305.0  #center of the mask in x
        Y_CENTER = 0.0    #center of the mask in y
        #set the fits values from autoslit that are used in the correction
        A=[2.0,  0.99476227,0.00728125,0.45412825E-6, -0.40955557E-5,0.96690643E-6]
        B=[1.0, -0.00001856, 0.99906324 ]

        #Convert mm to CCD pixels. (Approximate, but as long as you
        #use exactly the same formula for inversion, that is OK.)
        XCCD = CCD_SIDE / 2.0 + (X_CENTER - XIN) / CCD_SCALE
        YCCD = CCD_SIDE / 2.0 - (Y_CENTER - YIN) / CCD_SCALE

        XCCD_OUT = A[0] + A[1] * XCCD + A[2] * YCCD + A[3] * XCCD * XCCD + A[4] * YCCD * YCCD + A[5] * XCCD * YCCD
        YCCD_OUT = B[0] + B[1] * XCCD + B[2] * YCCD

        #Now convert back to mm on the slitmask.

        XOUT = X_CENTER - (XCCD_OUT - CCD_SIDE / 2.0) * CCD_SCALE
        YOUT = Y_CENTER + (YCCD_OUT - CCD_SIDE / 2.0) * CCD_SCALE
        return(XOUT-XIN)
#=================================================================================================================================
    #create the routine for the astrometry correction
    def x_astrometry_correction_calc_grid(XIN,YIN):
        """
        Function to calculate the distortion correction in the X position for a range of mask positions.
        This will be passed to the linear interpolator.

        Args:
            XIN: array
                1D array of X positions on mask

            YIN: array
                1D array of Y positions on mask

        Returns:
            2D Array of the distortion correction values for the X position.
        """

        # Constants
        CCD_SIDE = 2048.0
        CCD_SCALE = 0.15767
        X_CENTER = 305.0  #center of the mask in x
        Y_CENTER = 0.0    #center of the mask in y
        A=[2.0,  0.99476227,0.00728125,0.45412825E-6, -0.40955557E-5,0.96690643E-6]
        B=[1.0, -0.00001856, 0.99906324 ]

        # Convert mm to CCD pixels. (Approximate, but as long as you
        # use exactly the same formula for inversion, that is OK.)
        XCCD = CCD_SIDE / 2.0 + (X_CENTER - XIN[:, None]) / CCD_SCALE
        YCCD = CCD_SIDE / 2.0 - (Y_CENTER - np.swapaxes(YIN, 0, 1)[None, :]) / CCD_SCALE

        XCCD_OUT = A[0] + A[1] * XCCD + A[2] * YCCD + A[3] * XCCD**2 + A[4] * YCCD**2 + A[5] * XCCD * YCCD
        YCCD_OUT = B[0] + B[1] * XCCD + B[2] * YCCD

        # Now convert back to mm on the slitmask.
        XOUT = X_CENTER - (XCCD_OUT - CCD_SIDE / 2.0) * CCD_SCALE
        YOUT = Y_CENTER + (YCCD_OUT - CCD_SIDE / 2.0) * CCD_SCALE

        corrections = XOUT - XIN[:, None]
        return corrections

#=================================================================================================================================
    #create the routine for the astrometry correction
    def y_astrometry_correction_calc(XIN,YIN):
        """
        Function to calculate the distortion correction in the Y position for a range of mask positions

        Args:
            XIN: array
                1D array of X positions on mask

            YIN: array
                1D array of Y positions on mask

        Returns:
            Array of the distortion correction values for the Y position.
        """
        #set some constants
        CCD_SIDE = 2048.0
        CCD_SCALE = 0.15767
        X_CENTER = 305.0  #center of the mask in x
        Y_CENTER = 0.0    #center of the mask in y
        #set the fits values from autoslit that are used in the correction
        A=[2.0,  0.99476227,0.00728125,0.45412825E-6, -0.40955557E-5,0.96690643E-6]
        B=[1.0, -0.00001856, 0.99906324 ]

        #Convert mm to CCD pixels. (Approximate, but as long as you
        #use exactly the same formula for inversion, that is OK.)
        XCCD = CCD_SIDE / 2.0 + (X_CENTER - XIN) / CCD_SCALE
        YCCD = CCD_SIDE / 2.0 - (Y_CENTER - YIN) / CCD_SCALE

        XCCD_OUT = A[0] + A[1] * XCCD + A[2] * YCCD + A[3] * XCCD * XCCD + A[4] * YCCD * YCCD + A[5] * XCCD * YCCD
        YCCD_OUT = B[0] + B[1] * XCCD + B[2] * YCCD

        #Now convert back to mm on the slitmask.

        XOUT = X_CENTER - (XCCD_OUT - CCD_SIDE / 2.0) * CCD_SCALE
        YOUT = Y_CENTER + (YCCD_OUT - CCD_SIDE / 2.0) * CCD_SCALE

        return(YOUT-YIN)
#=================================================================================================================================
    #create the routine for the astrometry correction
    def y_astrometry_correction_calc_grid(XIN,YIN):
        """
        Function to calculate the distortion correction in the Y position for a range of mask positions.
        This will be passed to the linear interpolator.

        Args:
            XIN: array
                1D array of X positions on mask

            YIN: array
                1D array of Y positions on mask

        Returns:
            2D Array of the distortion correction values for the Y position.
        """

        # Constants
        CCD_SIDE = 2048.0
        CCD_SCALE = 0.15767
        X_CENTER = 305.0  #center of the mask in x
        Y_CENTER = 0.0    #center of the mask in y
        A=[2.0,  0.99476227,0.00728125,0.45412825E-6, -0.40955557E-5,0.96690643E-6]
        B=[1.0, -0.00001856, 0.99906324 ]

        # Convert mm to CCD pixels. (Approximate, but as long as you
        # use exactly the same formula for inversion, that is OK.)
        XCCD = CCD_SIDE / 2.0 + (X_CENTER - XIN[:, None]) / CCD_SCALE
        YCCD = CCD_SIDE / 2.0 - (Y_CENTER - np.swapaxes(YIN, 0, 1)[None, :]) / CCD_SCALE

        XCCD_OUT = A[0] + A[1] * XCCD + A[2] * YCCD + A[3] * XCCD**2 + A[4] * YCCD**2 + A[5] * XCCD * YCCD
        YCCD_OUT = B[0] + B[1] * XCCD + B[2] * YCCD

        # Now convert back to mm on the slitmask.
        XOUT = X_CENTER - (XCCD_OUT - CCD_SIDE / 2.0) * CCD_SCALE
        YOUT = Y_CENTER + (YCCD_OUT - CCD_SIDE / 2.0) * CCD_SCALE

        corrections = YOUT - np.swapaxes(YIN, 0, 1)[None, :]
        return(corrections)
    corr_x_range = x_range+x_astrometry_correction_calc(x_range,y_range)
    corr_y_range = y_range+y_astrometry_correction_calc(x_range,y_range)
    ref_data = x_astrometry_correction_calc_grid(x_grid, y_grid)
    y_ref_data = y_astrometry_correction_calc_grid(x_grid,y_grid)
    #call the actual interpolation function
    interp = RegularGridInterpolator((corr_x_range, corr_y_range), ref_data[:, :, 0],method='linear',bounds_error=True)
    yinterp = RegularGridInterpolator((corr_x_range,corr_y_range), y_ref_data[:, :, 0],method='linear',bounds_error=True)

#=================================================================================================================================
    #apply the correction to the X,Y slit positions
    # for i in range(len(data['X'])):
    #     import pdb; pdb.set_trace()
    #     data['X'][i] -= interp([data['X'][i],data['Y'][i]])
    #     data['Y'][i] -= yinterp([data['X'][i],data['Y'][i]])
    data['X'] -= interp(np.array([data['X'], data['Y']]).T)
    data['Y'] -= yinterp(np.array([data['X'], data['Y']]).T)
    #return the astropy results table
    return(data)
