"""
Module to construct a new mask FITS file from autoslit output .file3.
Writes a mask file in the structure of the UCO/Lick archive mask files with
slit locations and basic mask design information.
"""

from astropy.table import Table,vstack
from shutil import copyfile
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
import pathlib
import numpy as np

def lines_that_equal(line_to_match, fp):
    """
    Function to search for the lines in a file that match a given string
    Args:
        line_to_match: str
            string to exactly match in set of lines

        fp: str
            content of lines to search

    Returns:
        matches: array
            indices of lines that mactch line_to_match
    """
    matches = []
    for i in range(len(fp)):
        line = fp[i]
        if line == line_to_match:
            matches.append(i)
    return(matches)

def gen_from_auto(autofile):
    """
    Generate a FITS file similar to the UCO/Lick archival masks from the autoslit
    output file. Note that the mask design date is a placeholder meant to trigger whether the
    ADC was used for caluclations in the main code.

    Args:
        autofile: str
            Path to autoslit output .file3 for mask

    Returns:
        None
    """

    dir_loc = str(pathlib.Path(__file__).parent.resolve())

    #read in the autoslit output file
    auto_file = open(autofile+'.file3')

    content = auto_file.readlines()

    #need the center of the mask, epoch, position angle
    angle = content[26].split()[4]
    epoch = content[28].split()[-2]
    ra0 = content[58].split()[3]+content[58].split()[4]+content[58].split()[5]
    dec0 = content[59].split()[3]+content[59].split()[4]+content[59].split()[5]+content[59].split()[6]
    coords = SkyCoord(ra0,dec0,frame='fk5',equinox='J'+epoch)
    ra0 = coords.ra.deg
    dec0 = coords.dec.deg
    date = Time.strptime(content[2].split()[4]+' '+content[2].split()[3]+' '+content[2].split()[6],'%d %b %Y')

    #read in the X_mask,Y_mask slit positions

    slit_start_text = 'newrow\n'
    slit_start = lines_that_equal(slit_start_text,content)
    slit_pos = Table(names=['slitX1','slitY1','slitX2','slitY2',
                            'slitX3','slitY3','slitX4','slitY4'],
                            dtype=(float,float,float,float,float,float,float,float))

    for i in range(len(slit_start)-1):
        ind = slit_start[i]

        Z = []
        j = ind+1
        while content[j] != slit_start_text:
            # This variable holds both X and Y positions:
            Z.extend(content[j].split()[0:2])
            j+=1

        # The last vertex is a repeat of the first. Remove it.
        Z = Z[:len(Z)-2]

        # Test that we have only 4 vertices. These are the standard slits.
        if len(Z) == 8:
            slit_pos.add_row(Z)


    #write the new fits file with the same structure as the UCO/Lick archive
    copyfile(dir_loc+'/blank_template.fits', autofile+'.fits')
    new_file = fits.open(autofile+'.fits',mode='update')

    #write the design information to the fits file
    mask_design = Table(new_file['MaskDesign'].data)
    mask_design.add_row()
    mask_design['RA_PNT'] = ra0
    mask_design['DEC_PNT'] = dec0
    mask_design['PA_PNT'] = angle
    mask_design['EQUINPNT'] = epoch
    mask_design['RADEPNT'] = 'fk5'
    mask_design['DesDate'] = date.to_value('iso',subfmt='date')
    mask_design['DesNslit'] = len(slit_start)

    new_file['MaskDesign'] = fits.BinTableHDU(mask_design,header = new_file['MaskDesign'].header)

    #add the slit positions and unique dSlitId values in the bluslits table
    bluslits = Table(new_file['BluSlits'].data)
    bluslits = vstack([bluslits,slit_pos])
    bluslits['dSlitId'] = np.arange(len(bluslits))
    new_file['BluSlits'] = fits.BinTableHDU(bluslits,header = new_file['BluSlits'].header)

    new_file.flush()
