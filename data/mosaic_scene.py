'''
####    MOSAIC CONSTRUCTION FROM GUITARRA OUTPUT IMAGES    ####


Sandro Tacchella, sandro.tacchella@cfa.harvard.edu
November 27, 2019


This script constructs a mosaic from a set of input images and
corresponding flats. The final mosaic image is a weighted-mean
combination of individual exposures. Due to not flagged CR, an
additional step for masking CR has been implemented.


Steps:
    1.)  Collect images and make copies of them.
    2.)  Perform flat fielding.
    3.)  Construct median-combined mosaic (i.e. without CR).
    4.)  Project this mosaic back to the individual exposures.
    5.)  Perform sigma clipping to get detector artefacts that
         have not been included in the uncertainty images
         (i.e. CR).
    6.)  Construct the weight maps, taking into account the
         the sigma clipped images.
    7.)  Construct final mosaics:
            image = (sum img_i*wht_i) / (sum wht_i)
                  = average( img_i*wht_i ) / average( wht_i )
            with wht_i = 1 / unc_i^2
            err = sqrt( 1/N sum(err^2) )
                = sqrt( 1/N sum(1/wht) )
                = sqrt( average(1/wht) )
             => wht = 1/err^2
                    = 1 / average(1/wht)


Possible improvements in the future:

    - background subtraction and error propagation
    - optimize pixel scale of mosaic
    - shrinking of pixels before drizzeling?
    - chung data into dither patches (solve memory issue with large images)
    - specify working directory work_dir='TBD' when making mosaic


Example run:

python /Users/sandrotacchella/ASTRO/JWST/mini_data_challenge/mosaic_scene.py --filter 'F115W' --header '/Users/sandrotacchella/ASTRO/JWST/mini_data_challenge/data_challenge_header.hdr' --path_slp_data '/Volumes/Tacchella/Work/Postdoc/JWST_GTO/2019-mini-challenge/Step_Two_Slopes_withCR_removal/' --path_flat '/Volumes/Tacchella/Work/Postdoc/JWST_GTO/2019-mini-challenge/guitarra_median_sky_flats/' --path_mosaic '/Volumes/Tacchella/Work/Postdoc/JWST_GTO/2019-mini-challenge/mosaic/' --sigma_clip 5.0

# filter_in = 'F115W'
# header_file = '/Users/sandrotacchella/ASTRO/JWST/mini_data_challenge/data_challenge_header.hdr'
# path_slp_img = '/Volumes/Tacchella/Work/Postdoc/JWST_GTO/2019-mini-challenge/Step_Two_Slopes_withCR_removal/'
# path_flat = '/Volumes/Tacchella/Work/Postdoc/JWST_GTO/2019-mini-challenge/guitarra_median_sky_flats/'
# path_mosaic = '/Volumes/Tacchella/Work/Postdoc/JWST_GTO/2019-mini-challenge/mosaic/'
# sigma_clipping_threshold = 5.0

'''

# --------------
# import modules
# --------------

import argparse
import os
import shutil
import glob
import numpy as np

from astropy.io import fits
import montage_wrapper as montage
import sep


# --------------
# read command line arguments
# --------------

parser = argparse.ArgumentParser()
parser.add_argument("--filter", type=str, help="filter")
parser.add_argument("--header", type=str, help="header file")
parser.add_argument("--path_slp_data", type=str, help="path of slp images")
parser.add_argument("--path_flat", type=str, help="path of flat images")
parser.add_argument("--path_mosaic", type=str, help="path to mosaic folder")
parser.add_argument("--sigma_clip", type=float, help="sigma clipping threshold")
args = parser.parse_args()

filter_in = args.filter
header_file = args.header
path_slp_img = args.path_slp_data
path_flat = args.path_flat
path_mosaic = args.path_mosaic
sigma_clipping_threshold = args.sigma_clip


# ------------
# get images and move to new folder
# ------------

list_img = glob.glob(os.path.join(path_slp_img, '*' + filter_in + '*.slp.fits'))

try:
    shutil.rmtree(os.path.join(path_mosaic, filter_in))
except:
    print("Can't delete work tree; probably doesn't exist yet")

os.makedirs(os.path.join(path_mosaic, filter_in))

print('copying files...')

for f in list_img:
    shutil.copy(f, os.path.join(path_mosaic, filter_in))

list_img = glob.glob(os.path.join(path_mosaic, filter_in, '*.slp.fits'))


# --------------
# flat-fielding
# --------------

print('flat-fielding...')

for ii in range(len(list_img)):
    sca = list_img[ii].split('/')[-1].split('_')[-2]
    flat_hdul = fits.open(path_flat + 'median_sky_' + sca + '_' + filter_in + '.fits')
    flat_img = flat_hdul[0].data
    with fits.open(list_img[ii], mode='update') as hdul:
        hdul[0].data[0] = hdul[0].data[0]/flat_img
        hdul[0].data[1] = hdul[0].data[1]/flat_img
        hdul.flush()
        hdul.close()


# ----------------------------------
# create median mosaic
# ----------------------------------

print('create median mosaic...')

montage.mosaic(os.path.join(path_mosaic, filter_in), os.path.join(path_mosaic, filter_in, 'mosaic_median'), combine='median', background_match=False)


# ----------------------------------
# generate weighted images
# ----------------------------------

print('generate weighted images:')

for ii in range(len(list_img)):
    print(list_img[ii])
    # project image back to the individiual frames for CR identitication
    header = fits.getheader(list_img[ii])
    header.totextfile(list_img[ii].split('.')[0] + '.hdr', overwrite=True)
    montage.mProject_auto(in_image=os.path.join(path_mosaic, filter_in, 'mosaic_median', 'mosaic.fits'),
                          out_image=list_img[ii].split('.')[0] + '.median_depro.fits',
                          template_header=list_img[ii].split('.')[0] + '.hdr')
    # load median image
    median = fits.getdata(list_img[ii].split('.')[0] + '.median_depro.fits')
    # load image: signal and signal uncertainty
    hdul = fits.open(list_img[ii])
    hdr = hdul[0].header
    signal = hdul[0].data[0]
    signal_unc = hdul[0].data[1]
    hdul.close()
    # shape check and expansion if necessary (edge effects)
    shape_img = signal.shape
    shape_pro = median.shape
    diff_shape = np.array(signal.shape) - np.array(median.shape)
    if (diff_shape[0] > 0):
        median = np.vstack([median, np.nan*np.zeros(diff_shape[0], signal.shape[1])])
    if (diff_shape[1] > 0):
        median = np.hstack([median, np.nan*np.zeros((diff_shape[1], signal.shape[0])).T])
    # construct sigma image
    sigma = np.abs(signal-median)/signal_unc
    # load dia image and get CR
    dia = fits.getdata(os.path.join(path_slp_img, list_img[ii].split('.')[0].split('/')[-1] + '.dia.fits'))[0].astype(int)
    cr = (dia & 2048) == 2048
    # generate sigma mask
    mask = np.zeros(signal.shape)
    mask[(sigma > sigma_clipping_threshold) | cr] = 1.0
    hdu_mask = fits.PrimaryHDU(mask)
    hdu_mask.writeto(list_img[ii].split('.')[0] + '.cr_mask.fits', overwrite=True)
    # construct weight image: wht = 1 / unc^2
    wht = 1.0 / np.power(signal_unc, 2)
    # set CR flag to 0 in wht image
    wht[mask == 1.0] = 0.0
    wht[np.isnan(wht)] = 0.0
    wht[np.isnan(signal)] = 0.0
    # construct signal x wht image
    signal_weight = signal * wht
    signal_weight[np.isnan(signal_weight)] = 0.0
    # save images
    hdu_signal_weight = fits.PrimaryHDU(data=signal_weight, header=hdr)
    hdu_signal_weight.writeto(list_img[ii].split('.')[0] + '.signalwht.fits', overwrite=True)
    hdu_weight = fits.PrimaryHDU(data=wht, header=hdr)
    hdu_weight.writeto(list_img[ii].split('.')[0] + '.wht.fits', overwrite=True)
    hdu_weight_ing = fits.PrimaryHDU(data=1.0/wht, header=hdr)
    hdu_weight_ing.writeto(list_img[ii].split('.')[0] + '.wht_inv.fits', overwrite=True)


# ----------------------------------
# create mosaic
# ----------------------------------

# get list of images to combine

list_img_sigwht = glob.glob(os.path.join(path_mosaic, filter_in, '*.signalwht.fits'))
list_img_wht = glob.glob(os.path.join(path_mosaic, filter_in, '*.wht.fits'))
list_img_wht_inv = glob.glob(os.path.join(path_mosaic, filter_in, '*.wht_inv.fits'))

# place images to combine into new folders

os.mkdir(os.path.join(path_mosaic, filter_in, 'mosaic_mean'))
os.mkdir(os.path.join(path_mosaic, filter_in, 'mosaic_wht'))
os.mkdir(os.path.join(path_mosaic, filter_in, 'mosaic_wht_inv'))

for f in list_img_sigwht:
    shutil.copy(f, os.path.join(path_mosaic, filter_in, 'mosaic_mean'))

for f in list_img_wht:
    shutil.copy(f, os.path.join(path_mosaic, filter_in, 'mosaic_wht'))

for f in list_img_wht_inv:
    shutil.copy(f, os.path.join(path_mosaic, filter_in, 'mosaic_wht_inv'))


# average images with montage

montage.mosaic(os.path.join(path_mosaic, filter_in, 'mosaic_mean'), os.path.join(path_mosaic, filter_in, 'mosaic_mean', 'mosaic_mean'), combine='mean', header=header_file, background_match=False)
montage.mosaic(os.path.join(path_mosaic, filter_in, 'mosaic_wht'), os.path.join(path_mosaic, filter_in, 'mosaic_wht', 'mosaic_wht'), combine='mean', header=header_file, background_match=False)
montage.mosaic(os.path.join(path_mosaic, filter_in, 'mosaic_wht_inv'), os.path.join(path_mosaic, filter_in, 'mosaic_wht_inv', 'mosaic_wht_inv'), combine='mean', header=header_file, background_match=False)


# ----------------------------------
# finalize mosaic
# ----------------------------------

# make new director

final_dir = os.path.join(path_mosaic, filter_in + '_final')

try:
    shutil.rmtree(final_dir)
except:
    print("Can't delete work tree; probably doesn't exist yet")

os.mkdir(final_dir)


# load mean images

hdul_mean = fits.open(os.path.join(path_mosaic, filter_in, 'mosaic_mean', 'mosaic_mean', 'mosaic.fits'))
img_mean = hdul_mean[0].data
hdul_wht = fits.open(os.path.join(path_mosaic, filter_in, 'mosaic_wht', 'mosaic_wht', 'mosaic.fits'))
img_wht = hdul_wht[0].data
hdul_wht_inv = fits.open(os.path.join(path_mosaic, filter_in, 'mosaic_wht_inv', 'mosaic_wht_inv', 'mosaic.fits'))
img_wht_inv = hdul_wht_inv[0].data


# construct final outputs: weighted-mean image, wht image, and err image

img_final = img_mean/img_wht

hdu_final = fits.PrimaryHDU(data=img_final, header=hdul_mean[0].header)
hdu_final.writeto(os.path.join(final_dir, filter_in + '.fits'))

img_wht_final = 1.0/img_wht_inv

hdu_final_wht = fits.PrimaryHDU(data=img_wht_final, header=hdul_mean[0].header)
hdu_final_wht.writeto(os.path.join(final_dir, filter_in + '_wht.fits'))

hdu_final_err = fits.PrimaryHDU(data=1.0/np.sqrt(img_wht_final), header=hdul_mean[0].header)
hdu_final_err.writeto(os.path.join(final_dir, filter_in + '_err.fits'))


# ----------------------------------
# background subtraction with SEP
# ----------------------------------

# measure a spatially varying background on the image

bkg = sep.Background(img_final, bw=80, bh=80, fw=5, fh=5)

# get a "global" mean and noise of the image background:

print('global mean bkg', bkg.globalback)
print('global rms bkg', bkg.globalrms)

# evaluate background as 2-d array, same size as original image

bkg_image = bkg.back()

# save background and background subtracted image

hdu_bkg = fits.PrimaryHDU(data=bkg_image, header=hdul_mean[0].header)
hdu_bkg.writeto(os.path.join(final_dir, filter_in + '_bkg.fits'), overwrite=True)

hdu_final_bkg = fits.PrimaryHDU(data=img_final-bkg_image, header=hdul_mean[0].header)
hdu_final_bkg.writeto(os.path.join(final_dir, filter_in + '_bkgsub.fits'), overwrite=True)

