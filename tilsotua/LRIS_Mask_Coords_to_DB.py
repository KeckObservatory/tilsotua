"""
Main Module containing the function to calculate sky positions of LRIS Slitmasks.


refraction_correction: Function used to calculate the refraction correction at a
                        mask position

xytowcs: Main tilsotua function to be called to calculate mask positions with archival
         mask files or autoslit output files. Returns the sky positions of the slitmasks
         in mask fits file, DS9 region file, and CSV file.
"""
import psycopg2

import numpy as np
from astropy.coordinates import SkyCoord, ICRS, Galactic, FK4, FK5
from astropy.table import Table,Column,join,vstack
from astropy.io import ascii,fits
from astropy.time import Time
import astropy.units as u

import tilsotua.find_shifts as fs
import tilsotua.refractioncorrection as ref
import tilsotua.astrometrycorrection as ac
import tilsotua.use_autoslit_input as autoin

from shutil import copyfile

# Keck
import tilsotua.logger_utils as log_fun

log = log_fun.configure_logger('/kroot/var/log/slitmask/tilsotua/')


def insert_lris_db(data_input_name:str,output_file:str,design_id:int):
    """
    Run the insertion of object data into the database.

    Args:
        data_input_name: str
            Path to the input FITS file or autoslit-produced
            ".file3" file.

        output_file: str
            Name of the output file you'd
            like to generate. DO NOT INCLUDE FILE EXTENSIONS
            like .fits or .csv. e.g. "output_file".

    Returns:
        None
    """
    log.info(f'Running insertion of object data for design id: {design_id}, '
             f'{data_input_name}, {output_file}.')

    # First, generate the RA/Dec positions
    data_table = xytowcs(data_input_name, output_file)

    # Next, insert slit positions into the objects table
    object_id_pairs = insert_objects_table(data_table)
    if not object_id_pairs:
        log.error('Error during insertion,  no object pairs found.')
        return 0

    # insert into the slitobjmap table
    success = insert_slitobjmap(object_id_pairs, design_id)

    return success


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


def xytowcs(data_input_name:str,output_file:str):
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

    if data_ext== 'file3':
        autoin.gen_from_auto(data_input_name)

    #set up some constants
    rpd = np.pi/180. #radians per degree
    rpas = rpd/3600. #radians per arcsec
    mask_angle = 8.06 * rpd #angle of mask to the focal plane
    bend = 1.94 * rpd #angle of the bend in the mask
    x_center = 305. #point treated as the center on the ccd

    #  catalog_keyword = 'panstarrs'   #uncomment the desired catalog to be used
    catalog_keyword = 'gaia'

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
    log.info(f'Rotation Angle is selected at: {rot_angle}')

    #get the reference ra and dec
    #assume this to be the center of the image
    ra0,dec0,equ = mask_design['RA_PNT'], mask_design['DEC_PNT'],str(mask_design['EQUINPNT'])

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
        log.info('ADC Correction Used'.center(50, '-'))
    else:
        scale = 0.7253 #scale of mask in mm/arcsec, not corrected for ADC in use
        log.info('NO ADC Correction Used'.center(50, '-'))
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
    log.info(f'Reference Coordinates set at: ({ra0},{dec0})')

    #read in the slit data
    log.info('Reading in Slit Data'.center(50, '-'))

    bluslits = hdu['BluSlits'].data

    data = Table()
    data['X'] = np.array([bluslits['slitX1'], bluslits['slitX2'],
                          bluslits['slitX3'], bluslits['slitX4']]).T.flatten()
    data['Y'] = np.array([bluslits['slitY1'], bluslits['slitY2'],
                          bluslits['slitY3'], bluslits['slitY4']]).T.flatten()

    #The data from the html mask files is in the milling machine coordinate system. We have to convert to the mask coordinate
    #system before working with the data

    x_mill = data['X'].copy()
    y_mill = data['Y'].copy()

    data['X'] = y_mill + 172.7
    data['Y'] = -x_mill + 177.8

    x_input = data['X']
    y_input = data['Y']

    #apply the astrometry correction to the x,y values in the mask frame
    log.info('Applying Distortion Correction'.center(50, '-'))
    data = ac.astrometry_calc(data,ra0,dec0)

    #add columns to data table to eventually hold the calculated RA and Dec for each object
    data['Calc_RA'] = 0.
    data['Calc_Dec'] = 0.

    #=====================================================================================================================================

    #calculate the RA,Dec from mask coordinates

    #take theta to negative theta (internal angle opposite to recorded position angle)
    theta = (-rot_angle) * rpd

    log.info('Completing Inverse Gnomic Projection'.center(50, '-'))
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
    log.info('Calculating Refraction Lookup Table'.center(50, '-'))
    data = ref.refraction_calc(data,racenter*rpd,deccenter*rpd)

    temp1 = SkyCoord(ra=data['Calc_RA'], dec=data['Calc_Dec'], unit='deg', frame=FK5, equinox='J'+equ)
    temp_updated = temp1.transform_to(FK5(equinox='J2000'))
    data['Calc_RA'] = temp_updated.ra.deg
    data['Calc_Dec'] = temp_updated.dec.deg

    #====================================================================================================================================
    #calculate the average offset for the dataset and refraction correction
    log.info('Calculating Mask Shift'.center(50, '-'))
    # data,x_centers,y_centers,ra_shifted_centers,dec_shifted_centers,catalog_obj_ra,catalog_obj_dec,objects_ra,objects_dec = fs.get_shift(data,theta,catalog_keyword,output_file,ref_system,racenter,deccenter,adcuse,log)
    data,x_centers,y_centers,ra_shifted_centers,dec_shifted_centers,catalog_obj_ra,catalog_obj_dec,objects_ra,objects_dec = (fs.get_shift(data,catalog_keyword,ref_system,log))

    try:
        hdu['DesiSlits'].data['slitRA'] = data['RA_Center'][::4]
        hdu['DesiSlits'].data['slitDec'] = data['Dec_Center'][::4]
    except:
        desislits = Table(hdu['DesiSlits'].data)
        cen_slits = Table([data['RA_Center'][::4],data['Dec_Center'][::4]],names = ['slitRA','slitDec'])
        desislits = vstack([desislits,cen_slits])
        desislits['dSlitId'] = np.arange(len(desislits))
        hdu['DesiSlits'] = fits.BinTableHDU(desislits,header = hdu['DesiSlits'].header)

    log.info('Adding Slitname and SlitType'.center(50, '-'))

    desi_data = Table(hdu['DesiSlits'].data)  # convert HDU to Table for easy access

    #----------------Adding Slit Names to CSV-----------------
    data = add_column_data(data, desi_data, 'SlitName')
    data = add_column_data(data, desi_data, 'slitTyp')
    # ----------------Adding Slit Names to CSV-----------------

    hdu.flush()

    if 'SlitName' in data.colnames:
        # Reorder columns: SlitName first, then the rest
        new_order = ['SlitName'] + [col for col in data.colnames if col != 'SlitName']
        data = data[new_order]

    return data


def add_column_data(data_table, header_data, key_name):
    """
    Add a column to the data table.

    Args:
        data_table ():
        header_data ():
        key_name ():

    Returns:
        data_table (): updated data table.
    """
    nrows = len(data_table)
    rows_per_slit = 4

    if key_name in header_data.colnames:
        # Repeat each slit name for the desired number of rows
        names = np.repeat(header_data[key_name], rows_per_slit)

        # Truncate or pad to match the length of data
        if len(names) > nrows:
            names = names[:nrows]
        elif len(names) < nrows:
            names = np.concatenate([names, [''] * (nrows - len(names))])

        data_table[key_name] = names
    else:
        log.error(f"{key_name} column not found in desislits.")

    return data_table


def insert_objects_table(obj_data):
    """
    Keck Function.

    Insert object data into the database.

    Add the slit coordinates to the objects table in the
    slitmask database.  The table is used to map the slit
    positions on the mask to celestial coordinates.

    The table has foreign key constraint on the table
    slitobjmap,  so this step is done prior to inserting
    the 'objects' into slitobjmap.  The slitobjmap table
    is used to map the 'objects' in the slit to the
    design slits (desislits).

    Args:
        object_data: Table
            Table containing object data to be inserted.

    Returns:
        object_pairs: list of tuples
            A list of the slitname to object id pair.

    """
    conn = get_connected()
    curse = conn.cursor()

    object_pairs = []
    insert_cnt = 0

    for row in obj_data:
        slit_name = row['SlitName']
        ra = row['RA_Center']
        dec = row['Dec_Center']
        obj_class = row['slitTyp']

        curse.execute("""
            SELECT objectid
            FROM objects
            WHERE OBJECT=%s AND RA_OBJ=%s AND DEC_OBJ=%s AND ObjClass=%s
            ORDER BY objectid DESC
            LIMIT 1;
        """, (slit_name, ra, dec, obj_class))

        existing = curse.fetchone()
        if existing:
            object_pairs.append((slit_name, existing[0]))
        else:
            curse.execute("""
                INSERT INTO objects (OBJECT, RA_OBJ, DEC_OBJ, RADECSYS, EQUINOX, MJD_OBS,
                                     mag, pBand, RadVel, MajAxis, ObjClass)
                VALUES (%s, %s, %s, 'FK5', 2000.0, 0, 0, 'X', 0, 0, %s)
                RETURNING objectid;
            """, (slit_name, ra, dec, obj_class))
            new_id = curse.fetchone()[0]
            object_pairs.append((slit_name, new_id))

            insert_cnt += curse.rowcount

    conn.commit()
    close_conn(conn, curse)

    log.info(f"Inserted {insert_cnt} new rows into objects table.")

    return object_pairs


def insert_slitobjmap(object_pairs, desid):
    """
    Keck Function.

    The slitobjmap table is used to map the slit ids between the
    design slits (desislits) table and the objects table.

    The table has foreign key references to the table slitobjmap.
    The dslitid is obtained from the desislits table.  To determine
    the slit dslitid,  the desid (user) and slitname (object table) are
    used.

    Args:
        object_pairs: list of tuples
            A list of the slitname to object id pair.

    Returns:
        None

    """
    conn = get_connected()
    curse = conn.cursor()

    slit_maps = []

    for slit_name, object_id in object_pairs:

        query =  """
            SELECT dslitid
            FROM desislits
            WHERE desid = %s AND slitname = %s
            ORDER BY dslitid
            LIMIT 1;
        """

        # get
        curse.execute(query, (desid, slit_name))
        result = curse.fetchone()
        if not result:
            continue

        dslitid = result[0]

        slit_maps.append(
            (
                desid,      # desid
                object_id,  # objectid
                dslitid,    # dslitid
                0.0,        # topdist
                0.0         # botdist
            )
        )

    insert_cnt = 0
    for record in slit_maps:
        curse.execute("""
            INSERT INTO slitobjmap (desid, objectid, dslitid, topdist, botdist)
            SELECT %s, %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM slitobjmap
                WHERE desid = %s
                  AND objectid = %s
                  AND dslitid = %s
                  AND topdist = %s
                  AND botdist = %s
            );
        """, record + record)
        insert_cnt += curse.rowcount

    conn.commit()
    close_conn(conn, curse)

    log.info(f"Inserted {insert_cnt} new rows into slitobjmap table.")

    return 1

def get_connected():
    conn = psycopg2.connect(host="localhost",
        database="metabase",
        user="dbadmin",
        password="slit_mask4u!"
    )

    return conn

def close_conn(conn, curse):
    conn.commit()
    curse.close()
    conn.close()




