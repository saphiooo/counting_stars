# Counting Stars

## project overview
- finding stars in real datasets to discover interesting things!

## todo
- fix qq plots in counting_stars_v5 (they look wonky)
- time series from Swift UVOT

## fits image practice + time series
- mainly following tutorials, did some experimentation

## counting stars in the horsehead nebula
(same file tackling the same problem, but different verions saved; v1 = day 1, v2 = day 2, etc.)
- v1: gradient analysis 
- v2: using a circle 
- v3: annulus aperture 
- v4: different kinds of annuli, tweaking the snr ratio calculations

## swift UVOT data
- extracted data, counted stars
- floodfill to isolate stars
- working on determining astronomical phenomenon in the data using time series

## annulus folder
- differently sized annuli that were used to test out the counting stars algorithm
- number in file name indicates total height of the image in pixels
  - annulus 22 has a smaller circular aperture (2 pixels across)
  - annulus 31 has a larger circular aperture (5 pixels across)
  
