from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord

from regions import CircleSkyRegion

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

eventFile = fits.open('/home/saphio/sw00094137009um2w1po_uf.evt.gz')
imgFile = fits.open('/home/saphio/sw00094137009u_sk.img.gz')

primaryHdu = imgFile[0]
imgHdu = imgFile[1]

wcs = WCS(imgHdu.header)

events = eventFile[1].data

events_clean = events[np.where((events.QUALITY == 0))]
## quality flag of 0 = good
print(f'removed {len(events) - len(events_clean)} events')

#plt.subplot(projection=wcs)

fig, ax = plt.subplots(figsize=(11, 12))
ax.set_xlabel('time')

## initial hist
histdata, x_bins, y_bins, img = plt.hist2d(events_clean.X, events_clean.Y, bins=1000, vmin=0, vmax=200)
plt.colorbar()

## slider
fig.subplots_adjust(left=0.25, bottom=0.25)

axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(
    ax=axtime,
    label='num events',
    valmin=0,
    valmax=len(events_clean['TIME']),
    valinit=len(events_clean['TIME']),
)

## update function
def update(val):
    _, _, _, _ = ax.hist2d(events_clean.X[:int(val)], events_clean.Y[:int(val)], bins=1000, vmin=0, vmax=200)
    fig.canvas.draw_idle()

## change when slider is updated
slider.on_changed(update)

plt.show()
