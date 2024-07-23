# Counting Stars

## project overview
- finding stars to discover interesting things!
- using data from the Swift UVOT (ultraviolet optical telescope), this project aims to analyze stars that flare in the UV to determine characteristics about their activity, such as frequency
- this can be used to inform conclusions about potential exoplanets around them

## todo
- fix qq plots in counting_stars_v5 (they look wonky)
- identift astronomical event in swift UVOT data

## replicating this repo
- install astropy in the command line: `sudo apt install python3-astropy`
- install regions in the command line: `sudo apt install python3-regions`
- clone as usual! once cloning, make the following changes:
- image files: accessible from the folder "annuli_imgs"; replace the file
read-in in the first cell under the "star finding" header
- run notebook from top to bottom; do *not* skip any cells

## project work: swift UVOT data
- extracted data, counted stars
- floodfill to isolate stars
- working on determining astronomical phenomenon in the data using time series
  - using SNR ratio in the time domain to determine out-of-the-ordinary signals
  - working on helper functions to easily access stars from different kinds of
  inputs, and visualize stars in a time series for sanity checks
- a visualization program is also included to visualize the histogram over time; could potentially be used later

## practice: fits image practice + time series
- mainly following tutorials, did some experimentation

## practice: counting stars in the horsehead nebula
(same file tackling the same problem, but different verions saved; v1 = day 1, v2 = day 2, etc.)
- v1: gradient analysis 
- v2: using a circle 
- v3: annulus aperture 
- v4: different kinds of annuli, tweaking the snr ratio calculations

## annulus folder
- differently sized annuli that were used to test out the counting stars algorithm
- number in file name indicates total height of the image in pixels
  - annulus 22 has a smaller circular aperture (2 pixels across)
    - also used in the swift UVOT program code
  - annulus 31 has a larger circular aperture (5 pixels across)
  
