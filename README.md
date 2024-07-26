# Counting Stars

## project overview
- finding stars to discover interesting things!
- using data from the Swift UVOT (ultraviolet optical telescope), this project aims to analyze stars that flare in the UV to determine characteristics about their activity, such as frequency
- this can be used to inform conclusions about potential exoplanets around them

## end goal
- a single clean python script that can take any Swift UVOT data and search for astronomical events of note

## todo
- fix qq plots in counting_stars_v5 (they look wonky)
- identify astronomical event in swift UVOT data
  - cleaned data may be buggy?
  - edge-handling may lead to false outliers
  - working out outliers

## replicating this repo
- clone the repo:
  ```
  git clone git@github.com:saphiooo/counting_stars.git
  ```
- install packages in the command line:
  ```
  sudo apt install python3-matplotlib 
  sudo apt install python3-astropy
  sudo apt install python3-regions
  sudo apt install python3-photutils
  sudo apt install python3-scipy
  sudo apt install python3-wget
  ```
- python files of the jupyter notebooks are given in the repo, however
you are also free to convert the notebooks to python files by yourself; type
the following in the command line:
  ```
  sudo apt install jupyter-nbconvert
  jupyter nbconvert --to script script swift_uvot_events.ipynb
  ```
- (alternatively, replace the file name with any other notebook name in this repo; however they are not of great importance to the main project)
- **PLEASE NOTE** that until project code is fully finished, the python version **is buggy**
- this saves the notebook as a python file in the current directory, which you can run with
  ```
  python3 swift_uvot_events.py
  ```
- note that many graphs and plots are being generated, and data filtering, despite running in polynomial time, is slow, thus the program is incredibly time-consuming (potentially several hours) 
- if using jupyter notebook, run the cells **in order** 
- necessary files are downloaded in the scripts; do not change file paths in the scripts
- the simple interactive visualization program can be run with
  ```
  python3 visualize_events.py
  ```
  but is slightly laggy due to large amount of data

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
- practicing extracting star data from an image, using the signal-to-noise
ratio

## annulus folder
- differently sized annuli that were used to test out the counting stars algorithm
- number in file name indicates total height of the image in pixels
  - annulus 22 has a smaller circular aperture (2 pixels across)
    - also used in the swift UVOT program code
  - annulus 31 has a larger circular aperture (5 pixels across)
  
