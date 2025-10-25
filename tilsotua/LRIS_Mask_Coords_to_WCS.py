"""
Main Module containing the function to calculate sky positions of LRIS Slitmasks.

Functions:
generate_object_cat: Populates ObjectCat, SlitObjMap and DesiSlit tables using the
                    .file1 autoslit output file and object file provided to autoslit
                    during mask creation.

refraction_correction: Function used to calculate the refraction correction at a
                        mask position

xytowcs: Main tilsotua function to be called to calculate mask positions with archival
         mask files or autoslit output files. Returns the sky positions of the slitmasks
         in mask fits file, DS9 region file, and CSV file.
"""
#import packages
import sys
import numpy as np
from astropy.coordinates import SkyCoord, ICRS, Galactic, FK4, FK5
from astropy.table import Table,Column,join,vstack
from astropy.io import ascii,fits
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt

import tilsotua.find_shifts as fs
import tilsotua.refractioncorrection as ref
import tilsotua.astrometrycorrection as ac
import tilsotua.create_quicklookplot as qlp
import tilsotua.write_ds9_file as w9f
import tilsotua.precessionroutine as pr
import tilsotua.use_autoslit_input as autoin

from shutil import copyfile
import warnings

def generate_object_cat(obj_file:str, file1:str, xy_map:Table,
                        mag_band:str='I')->Table:
    """
    Fill in the ObjectCat, SlitObjMap and DesiSlit tables in the
    output fits file.

    Args:
        obj_file: str
            Path to object file fed to AUTOSLIT

        file1: str
            Path to the autoslit object file

        xy_map: table
            RA and Dec slit positions

        mag_band: str, optional
            This is the band in which
            the object magnitudes are reported in the AUTOSLIT
            files. Unfortunately, this info must be manually input to
            this function as AUTOSLIT doesn't recieve/record it. Assumes
            I band by default.

    Returns:
        ObjectCat: table
            Object catalog.

        SlitObjMap: table
            Table mapping slits to catalog objects.

        DesiSlits: table
            Slit geometry parameter table. Does not
            contain slitRA/slitDec columns. Those are filled outside
            this function.
    """
    assert type(mag_band) == str, "Invalid band for magnitude"

    # Read in tables
    #objcols = ['Name', 'prior', 'Mag', '']
    objects = Table.read(obj_file, format="ascii")
    objects.rename_column('col1', 'Name')
    
    #generate the line information for the autoslit file
    auto_file = open(file1)
    content = auto_file.readlines()

    #find the header line in the autoslit file
    auto_header_start = autoin.lines_that_equal('#  X_STAR    Y_STAR       MIN_Y       MAX_Y    Percent   CCD_X   CCD_Y Prior     Name           Mag  #\n',content)
    file1_tab = Table.read(file1, format="ascii.commented_header", header_start=auto_header_start[0])
    file1_tab['index'] = np.arange(len(file1_tab))

    unique_names = np.unique(file1_tab['Name'])
    assert len(unique_names) == len(file1_tab), "There are non-unique object names in file1. Cannot assign object names from this file. Aborting."

    unique_objs = np.unique(objects['Name'])
    assert len(unique_objs) == len(objects), "There are non-unique object names in the obj file. Cannot uniquely match entries in file1. Aborting."

    # Join by object name
    joined_tab = join(file1_tab, objects, keys='Name',join_type='inner')
    joined_tab.sort('index') # Preserve the order in file1_tab
    # Get RA and Dec

    obj_ra = joined_tab['col4']+joined_tab['col5']/60+joined_tab['col6']/3600 # hours
    obj_ra *= 15 # Convert to degrees.
    obj_dec = joined_tab['col7']+(joined_tab['col8']/60+joined_tab['col9']/3600)*np.sign(joined_tab['col7']) # degs

    # Get ObjectClass. This is sketchy. The AUTOSLIT files don't contain this
    # info eplicitly. However, the Percent column value in .file1 is always 50.0
    # for stars
    obj_class = np.where(joined_tab['Percent']==50.0, 'Alignment_Star','Program_Target')

    # Initilize ObjectCat
    ObjectCat = Table()
    ObjectCat['OBJECT'] = joined_tab['Name']
    ObjectCat['ObjectId'] = np.arange(len(ObjectCat))
    ObjectCat['RA_OBJ'] = obj_ra
    ObjectCat['DEC_OBJ'] = obj_dec
    ObjectCat['RADESYS'] = ''
    ObjectCat['EQUINOX'] = joined_tab['col10']
    ObjectCat['MJD-OBS'] = 0.0 # No idea why but this is what's in the DEIMOS HDU
    ObjectCat['mag'] = joined_tab['Mag']
    ObjectCat['pBand'] = mag_band
    ObjectCat['RadVel'] = 0.0
    ObjectCat['MajAxis'] = 0.0
    ObjectCat['MajAxPA'] = 0.0
    ObjectCat['MinAxis'] = 0.0
    ObjectCat['MinAxPA'] = 0.0
    ObjectCat['PM_RA'] = 0.0
    ObjectCat['PM_Dec'] = 0.0
    ObjectCat['Parallax'] = 0.0
    ObjectCat['ObjectClass'] = obj_class
    ObjectCat['CatFilePK'] = 1 # No idea what this is. Again going with DEIMOS HDUs

    # Now work on the SlitObjMap table

    SlitObjMap = Table()
    SlitObjMap['ObjectId'] = ObjectCat['ObjectId']
    SlitObjMap['DesId'] = 1 # Dunno why.
    SlitObjMap['dSlitId'] = np.arange(len(SlitObjMap))

    alpha1 = xy_map['Calc_RA'][::4]
    delta1 = xy_map['Calc_Dec'][::4]

    alpha2 = xy_map['Calc_RA'][1::4]
    delta2 = xy_map['Calc_Dec'][1::4]

    alpha3 = xy_map['Calc_RA'][2::4]
    delta3 = xy_map['Calc_Dec'][2::4]

    alpha4 = xy_map['Calc_RA'][3::4]
    delta4 = xy_map['Calc_Dec'][3::4]

    coord_14 = SkyCoord((alpha1+alpha4)/2, (delta1+delta4)/2, unit="deg")
    coord_23 = SkyCoord((alpha2+alpha3)/2, (delta2+delta3)/2, unit="deg")
    coord_12 = SkyCoord((alpha1+alpha2)/2, (delta1+delta2)/2, unit="deg")
    coord_34 = SkyCoord((alpha4+alpha3)/2, (delta4+delta3)/2, unit="deg")

    objcoord = SkyCoord(obj_ra, obj_dec, unit="deg")
    bot_dist = objcoord.separation(coord_23)
    top_dist = objcoord.separation(coord_14)

    SlitObjMap['TopDist'] = top_dist.to('arcsec').value
    SlitObjMap['BotDist'] = bot_dist.to('arcsec').value

    slit_len = coord_14.separation(coord_23).to('arcsec').value
    slit_LPA = coord_14.position_angle(coord_23).to('deg').value

    slit_wid = coord_12.separation(coord_34).to('arcsec').value
    slit_WPA = coord_34.position_angle(coord_12).to('deg').value

    DesiSlits = Table()
    DesiSlits['dSlitId'] = SlitObjMap['dSlitId']
    DesiSlits['DesId'] = SlitObjMap['DesId']
    DesiSlits['SlitName'] = ['slit'+str(idx) for idx in range(len(DesiSlits))]
    DesiSlits['slitTyp'] = [entry[0] for entry in ObjectCat['ObjectClass']]
    DesiSlits['slitLen'] = slit_len
    DesiSlits['slitLPA'] = slit_LPA
    DesiSlits['slitWid'] = slit_wid
    DesiSlits['slitWPA'] = slit_WPA

    return ObjectCat, SlitObjMap, DesiSlits

def refraction_correction(ra0:float, dec0:float):
    """
    Given the RA and Dec of the pointing center for a mask,
    correct for refraction and return the delta in RA and Dec.

    Args:
        ra,dec: float,float
            Pointing coordinates

    Returns:
        dRa, dDec: float, float
            Delta in RA and Dec to be added
            to ra, dec for the correct pointing.
    """
    #Taken directly from autoslit
    rpd = np.pi/180. #radians per degree
    rph = rpd*15     #radians per hour
    rpas = rpd/3600. #radians per arcsec
    HA = 0.   #hour angle
    wave = 6000./10000  #wavelength in microns
    lat = 19.828 #keck latitude
    temperature = 0.
    pres = 486.  #atmos pressure
    ra0 = ra0*rpd
    dec0 = dec0*rpd
    H = HA*rph + ra0 - ra0
    cosz = np.sin(lat*rpd)*np.sin(dec0)+np.cos(lat*rpd)*np.cos(dec0)*np.cos(H)
    sinz = np.sqrt(1-(cosz**2))
    tanz = sinz/cosz
    if sinz != 0.:
        sinQ = np.cos(lat*rpd)*np.sin(H)/sinz
        cosQ = (np.cos(np.pi/2-lat*rpd)-cosz*np.cos(np.pi/2-dec0))/(sinz*np.sin(np.pi/2-dec0))
    else:
        sinQ = 0.
        cosQ = 0.
    WAVERS = 1. / (wave * wave)
    N = 64.328 + (29498.1/(146.-WAVERS)) + (255.4/(41.-WAVERS))
    N = 1.*10**-6 * N
    TCORR = 1. + 0.003661 * temperature
    NUM = 720.88 * TCORR
    N = N * ((pres * (1.+(1.049-0.0157*temperature) * 1.E-6 * pres)) / NUM)

    R = N * 206265. * tanz

    DA1 = R * sinQ * rpas / np.cos(dec0)
    DD1 = R * cosQ * rpas
    return DA1, DD1


def xytowcs(data_input_name:str,output_file:str,
            obj_file:str=None, file1:str=None,
            mag_band:str='I')->None:
    """
    Function to convert slit coordinates in the mask frame
    to equatorial coordinates (RA,Dec). Generates a CSV
    file and a FITS file with the RA, Dec information
    corresponding to each slit center. The output FITS
    file has the same structure as the input file but has
    the missing data filled in.

    Args:
        data_input_name: str
            Path to the input FITS file or autoslit-produced
            ".file3" file. The FITS file would be produced during the mask design
            ingestion process. i.e. the FITS file generated when AUTOSLIT .file3
            ascii files are fed mask submission webpage.  If the input is a .file3,
            a FITS file for the mask will be automatically generated.

        output_file: str
            Name of the output file you'd
            like to generate. DO NOT INCLUDE FILE EXTENSIONS
            like .fits or .csv. e.g. "output_file".

        obj_file: str, optional
            Path to the object catalog file that was used as an input to AUTOSLIT
            when generating the mask design files corresponding to data_input_name.

        file1: str, optional
            Path to the list of objects generated
            by AUTOSLIT. Has the extension of ".file1" by default. OBJ_FILE
            AND FILE1 ARE NECESSARY TO POPULATE OBJECT NAMES IN THE OUTPUT.

        autofile: str, optional
            Path to the autoslit output file
            (extension ".file3") to be used if the mask FITS file needs to
            be generated before anything else is done.

        mag_band: str, optional
            Filter band in which AUTOSLIT
            was fed object magnitudes.

    Returns:
        None
    """

    #If the data input file is a ".file3" autoslit output file, generate the mask
    #FITS file first
    data_input_name,data_ext = data_input_name.split('.')[0],data_input_name.split('.')[1]
    #data_ext = data_input_name.split('.')[1]
    if data_ext== 'file3':
        autoin.gen_from_auto(data_input_name)

    #set up some constants
    rpd = np.pi/180. #radians per degree
    rph = rpd*15     #radians per hour
    rpas = rpd/3600. #radians per arcsec
    mask_angle = 8.06 * rpd #angle of mask to the focal plane
    bend = 1.94 * rpd #angle of the bend in the mask
    x_center = 305. #point treated as the center on the ccd

    #  catalog_keyword = 'panstarrs'   #uncomment the desired catalog to be used
    catalog_keyword = 'gaia'
    #   catalog_keyword = 'custom'
    if catalog_keyword == 'custom':
        catalog_file = input('Catalog filename:')
    else:
        catalog_file = ''

    #================================================================================================================================

    #read in the data\
    #copy the original file to a new file that will be the one to
    # which the results are added

    copyfile(data_input_name+'.fits', output_file+'.fits')
    #Open the copied file
    hdu=fits.open(output_file+'.fits',mode='update')

    #obtain the rotation angle of the instrument
    mask_design = hdu['MaskDesign'].data[0]
    rot_angle = mask_design['PA_PNT']
    print('Rotation Angle is selected at:',rot_angle)

    #get the reference ra and dec
    #assume this to be the center of the image
    ra0,dec0,equ = mask_design['RA_PNT'], mask_design['DEC_PNT'],str(mask_design['EQUINPNT'])
    epoch = mask_design['EQUINPNT']

    #================================================================================================================================
    #Correct the center of the mask for refraction
    DA1, DD1 = refraction_correction(ra0, dec0)
    #set the center of the mask to the corrected value
    ra0 =  str(ra0 +  DA1/rpd)
    dec0 = str(dec0 + DD1/rpd)

    #Grab the date the mask was made
    creation_date = mask_design['DesDate']
    #Set the mask scale. This depends on whether the mask was made to be used with or without the ADC (ADC installed for B semester of 2007)
    if Time(creation_date) > Time('2007-08-01'):
        scale = 0.7253 *0.99857#scale of mask in mm/arcsec, corrected for ADC in use
        adcuse = 'post'
        print('--------------ADC Correction Used-----------------')
    else:
        scale = 0.7253 #scale of mask in mm/arcsec, not corrected for ADC in use
        print('--------------NO ADC Correction Used-----------------')
        adcuse = 'pre'
    x0 = x_center / scale

    ref_system = str(mask_design['RADEPNT']).lower()
    if ref_system == '':
        ref_system = 'fk5'
    #convert reference ra,dec to decimal degrees
    #correct for precession of coordiantes based on the equinox they are given in

    #set up SkyCoord object for the reference RA,Dec
    temp = SkyCoord(ra0+' '+dec0,frame=ref_system,unit=(u.deg,u.deg),equinox='J'+equ)
    ra0 = temp.ra.deg
    dec0 = temp.dec.deg
    racenter = temp.ra.deg
    deccenter = temp.dec.deg
    print('Reference Coordinates set at: (',ra0,',',dec0,')')

    #read in the slit data
    print('------------Reading in Slit Data-----------------')
    bluslits = hdu['BluSlits'].data

    data = Table()
    data['X'] = np.array([bluslits['slitX1'], bluslits['slitX2'],
                          bluslits['slitX3'], bluslits['slitX4']]).T.flatten()
    data['Y'] = np.array([bluslits['slitY1'], bluslits['slitY2'],
                          bluslits['slitY3'], bluslits['slitY4']]).T.flatten()

    #The data from the html mask files is in the milling machine coordinate system. We have to convert to the mask coordinate
    #system before working with the data
    #x_mask = y_mill + 172.7
    #y+mask = -x_mill +177.8

    x_mill = data['X'].copy()
    y_mill = data['Y'].copy()

    data['X'] = y_mill + 172.7
    data['Y'] = -x_mill + 177.8


    x_input = data['X']
    y_input = data['Y']

    #apply the astrometry correction to the x,y values in the mask frame
    print('-------------------Applying Distortion Correction-------------------')
    #astro_table = ac.astrometry_calc(data,ra0,dec0)
    data = ac.astrometry_calc(data,ra0,dec0)

    #add columns to data table to eventually hold the calculated RA and Dec for each object
    data['Calc_RA'] = 0.
    data['Calc_Dec'] = 0.

    #=====================================================================================================================================

    #calculate the RA,Dec from mask coordinates

    #take theta to negative theta (internal angle opposite to recorded position angle)
    theta = (-rot_angle) * rpd

    print('--------------------Completing Inverse Gnomic Projection--------------------')
    #correct the x and y coords for the bend and tilt in the mask
    x_prime = np.cos(mask_angle)*(x_input)
    y_prime = np.cos(bend)*(y_input)

    #take x,y to the eta and nu gnomic projection coordinates (rotation matrix)
    eta = (np.cos(theta)*x_prime-np.sin(theta)*y_prime)*rpas/scale
    nu =  (np.sin(theta)*x_prime+np.cos(theta)*y_prime)*rpas/scale
    #take eta and nu to ra and dec (standard already calculated inversion)
    rho = np.sqrt(eta**2+nu**2)
    c = np.arctan2(rho,1.)

    #have to correct the center coordiantes before inverse gnomic projectoin
    ra0 = ra0*rpd - (x0*np.cos(theta)*rpas/np.cos(dec0*rpd))
    dec0 = dec0*rpd - x0*np.sin(theta)*rpas

    #calculate the final ra and dec using the inverse gnomic projection
    ra_t = eta*np.sin(c)
    ra_b = (rho*np.cos(dec0)*np.cos(c))-(nu*np.sin(dec0)*np.sin(c))
    dec_f = np.cos(c)*np.sin(dec0)
    dec_s = (nu*np.sin(c)*np.cos(dec0))/rho
    RA = (ra0+np.arctan2(ra_t/ra_b,1.))/rpd
    Dec = np.arcsin(dec_f+dec_s)/rpd

    data['Calc_RA']=RA
    data['Calc_Dec']=Dec

    #===================================================================================================================================
    #Now apply refraction correction to each of the points on the mask
    print('-----------------Calculating Refraction Lookup Table------------------------')
    data = ref.refraction_calc(data,racenter*rpd,deccenter*rpd)

    #update the slit locations to J2000
    temp1 = SkyCoord(ra=data['Calc_RA'], dec=data['Calc_Dec'], unit='deg', frame=FK5, equinox='J'+equ)
    temp_updated = temp1.transform_to(FK5(equinox='J2000'))
    data['Calc_RA'] = temp_updated.ra.deg
    data['Calc_Dec'] = temp_updated.dec.deg

    #update the mask center to J2000 coordinates
    if equ[0:4] != '2000':
        print('UPDATED MASK CENTER TO J2000 COORDINATES')
    temp1 = SkyCoord(ra=hdu['MaskDesign'].data['RA_PNT'], dec=hdu['MaskDesign'].data['DEC_PNT'], unit='deg', frame=FK5, equinox='J'+equ)
    temp_updated = temp1.transform_to(FK5(equinox='J2000'))
    hdu['MaskDesign'].data['RA_PNT'] = temp_updated.ra.deg[0]
    hdu['MaskDesign'].data['DEC_PNT'] = temp_updated.dec.deg[0]
    hdu['MaskDesign'].data['EQUINPNT'] = '2000.0'  


    #====================================================================================================================================
    #calculate the average offset for the dataset and refraction correction
    print('----------------Calculating Mask Shift-----------------')
    data,x_centers,y_centers,ra_shifted_centers,dec_shifted_centers,catalog_obj_ra,catalog_obj_dec,objects_ra,objects_dec = fs.get_shift(data,theta,catalog_keyword,output_file,ref_system,racenter,deccenter,adcuse)
    #create the error plot
    print('-----------------Creating Quick Look Plot---------------')
    #try:
    qlp.create_qlp(data,catalog_obj_ra,catalog_obj_dec,objects_ra,objects_dec,output_file)
    #except:
    #    print('ISSUE WITH QUICK LOOK PLOT...CHECK RESULTS')

    w9f.create_ds9_file(data,rot_angle,scale,ref_system,output_file)
    #=====================================================================================================================================
    #update the fits file extension to include the calculated center positions of the slits

    try:
        hdu['DesiSlits'].data['slitRA'] = data['RA_Center'][::4]
        hdu['DesiSlits'].data['slitDec'] = data['Dec_Center'][::4]
    except:
        desislits = Table(hdu['DesiSlits'].data)
        cen_slits = Table([data['RA_Center'][::4],data['Dec_Center'][::4]],names = ['slitRA','slitDec'])
        desislits = vstack([desislits,cen_slits])
        desislits['dSlitId'] = np.arange(len(desislits))
        hdu['DesiSlits'] = fits.BinTableHDU(desislits,header = hdu['DesiSlits'].header)

    '''
    if data_ext == 'file3':
        desislits = Table(hdu['DesiSlits'].data)
        cen_slits = Table([data['RA_Center'][::4],data['Dec_Center'][::4]],names = ['slitRA','slitDec'])
        desislits = vstack([desislits,cen_slits])
        hdu['DesiSlits'] = fits.BinTableHDU(desislits,header = hdu['DesiSlits'].header)
    else:
        hdu['DesiSlits'].data['slitRA'] = data['RA_Center'][::4]
        hdu['DesiSlits'].data['slitDec'] = data['Dec_Center'][::4]
    '''

    # If additional information is given, also populate the ObjectCat and
    # SlitObjMap tables
    if obj_file and file1:
        ObjectCat, SlitObjMap, DesiSlits_part = generate_object_cat(obj_file, file1, data, mag_band)
        hdu['ObjectCat'] = fits.BinTableHDU(ObjectCat, header=hdu['ObjectCat'].header)
        hdu['SlitObjMap'] = fits.BinTableHDU(SlitObjMap, header=hdu['SlitObjMap'].header)
        for col in DesiSlits_part.colnames:
            hdu['DesiSlits'].data[col] = DesiSlits_part[col]
    else:
        warnings.warn("ObjectCat and SlitObjMap tables are not populated. Your output will not contain object names.",
                      category=UserWarning)
    hdu.flush()

    #write out the data and results to the output file
    ascii.write(data,output_file+'.csv',format='csv',overwrite=True)


    return
