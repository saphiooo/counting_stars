# Counting Stars: a Repository

* Status: Content Mostly Written

## Flare Finder: an Algorithm to Detect Near-Ultraviolet Flares in M-dwarfs in *Swift* UVOT Event Data 
### Project overview
M-dwarfs, which comprise 75% of the stellar population, are typically favorable for transiting exoplanets. They have cool effective temperatures, relatively low mass, and strong magnetic fields relative to their size. In addition, their stellar flares are observable across the EM spectrum from gamma rays to radio waves, and characterizing these flares is a key step in determining habitability of transiting exoplanets, especially in the UV ([Paudel et al.](https://arxiv.org/abs/2404.12310)).

However, to this point, flares are typically first detected in the optical wavelength, and then other telescopes are pointed at the source to measure the flare in other wavelengths, and only 28% of flares detectable in the NUV are not detectable optically. Flare Finder aims to detect flares in any inputted Swift UVOT data, without the need for optical data, reducing bias in observed flares.

You can read my paper by following these steps:

First, clone the repo with
  ```
  git clone git@github.com:saphiooo/counting_stars.git
  ```
Install necessary packages:
  ```
  sudo apt-get update -y
  sudo apt-get install texlive
  sudo apt-get install -y texlive-science
  sudo apt-get install texlive-bibtex-extra biber
  ```
Then run the commands
  ```
  cd counting_stars/flare_finder_paper
  biber main
  ```
Compile and view the document with
  ```
  pdflatex -interaction=nonstopmode main.tex
  evince main.pdf
  ```
If an error occurs with the references/bibliography, you may have to rerun `biber main`.

You can convert the slideshow to a PDF with
  ```
  soffice --headless --convert-to pdf flare_finder_slides.odp
  ```

Alternatively, you can also open the slideshow in Libreoffice with
  ```
  libreoffice --writer flare_finder_slides.odp
  ```
which will open the presentation directly in LibreOffice Impress (on Linux).

### Known Issues/Future Improvement
Known areas of improvement include:
- Distinguishing noise flares from real data
- Filtering real candidates for flares from false outliers
- Standard deviation concerns in signal-to-noise ratio calculation

### Replicating This Repository
First, clone the repo with
  ```
  git clone git@github.com:saphiooo/counting_stars.git
  ```
Then, install the necessary packages in the command line with
  ```
  sudo apt install python3-matplotlib 
  sudo apt install python3-astropy
  sudo apt install python3-regions
  sudo apt install python3-photutils
  sudo apt install python3-scipy
  sudo apt install python3-wget
  sudo apt install python3-pillow
  ```
Then enter the repository with
  ```
  cd counting_stars
  ```
and run the main Python file with
  ```
  python3 analyze_swift_uvot_events.py https://www.swift.ac.uk/archive/reproc/00094137009/uvot/event/sw00094137009um2w1po_uf.evt.gz https://www.swift.ac.uk/archive/reproc/00094137009/uvot/products/sw00094137009u_sk.img.gz
  ```
Alternatively, you may run the program on your own data. The first link should be to *Swift* UVOT event data, and the second should be to UVOT image data.

From there, the program will run on its own. Despite running in polynomial time, the data filtering and iteration over data will take some time. Take a coffee break.

### Using Other Functions
There are other functions included in the main python script that are not called, or can be called outside of automated processs.
These include:
- `visualizeStarTS()`: a method that visualizes an entire or part of a TimeSeries with certain x bounds, and marks the signal and background windows in green and red respectively.
  - Parameters: starNum (index or star name), starData (binned Astropy TimeSeries of star data), [xmin = minimum of x axis, xmax = maximum of x axis, windowStart = timestamp start of signal window, windowSize = length of signal window, customLim = True if xmin/xmax are specified, point = True to display the TimeSeries as non-connected points]
- `getSignals()`: get all SNRs for a star's data.
  - Parameters: signalWindow (length of signal window), starData (binned Astropy TimeSeries of star data), [printLog = True to print statistics of each measured time window when calculating ratios]
- `visualizeSignals()`: a method that displays a histogram of all calculated SNR ratios for a certain star, then fits a Gaussian distribution over the data (red) and marks the upper 5-standard-deviation bound.
  - Parameters: ratios (list of SNRs by timestamp), measuredTimestamps (list of all timestamps that were measured in order). These are the output of the getSignals() function.

### Other Scripts in the Repository
Some other jupyter notebooks and python scripts are included in the repository that detail earlier stages of the Flare Finder. You are welcome to experiment with them, though they mostly follow online tutorials and contain some of my own experimentation.

Although Python versions of all Jupyter notebooks are already included, you are free to convert them on your own, with
  ```
  sudo apt install jupyter-nbconvert
  jupyter nbconvert --to script script swift_uvot_events.ipynb
  ```
Alternatively, replace the file name with any other notebook name in this repo; however they are not of great importance to the main project. This saves the notebook as a python file in the current directory, which you can run with
  ```
  python3 swift_uvot_events.py
  ```
Moreover, I have included a simple interactive visualization program that can be run with
  ```
  python3 visualize_events.py
  ```
The testing for calculating the signal-to-noise ratio was done on an image of the [Horsehead Nebula](https://en.wikipedia.org/wiki/Horsehead_Nebula), which you are free to try out here:
  ```
  python3 counting_stars_v5.py
  ```

### The Annulus Folder (annuli_imgs)
This is a folder to host the images I used for differently sized annuli. These annulus apertures were used to test the
coutning stars algorithm. The number in the file name indicates the total height/width of the image in pixels, and the circular aperture inside has an average width of around three pixels across. Annulus 22 has a smaller circular aperture that is better at detecting faint or overlapping stars, whereas annulus 31 has a larger circular aperture for larger and more clearly-defined stars.
