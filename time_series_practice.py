#!/usr/bin/env python
# coding: utf-8

# # Time Series Practice

# ## Importing
# 

# In[1]:


from astropy.utils.data import get_pkg_data_filename
filename = get_pkg_data_filename('timeseries/kplr010666592-2009131110544_slc.fits')  


# In[2]:


from astropy.timeseries import TimeSeries
from astropy.timeseries import BoxLeastSquares


# In[3]:


ts = TimeSeries.read(filename, format='kepler.fits')


# In[4]:


print(ts)


# In[5]:


print(ts['cadenceno'])


# In[6]:


print(ts['time', 'sap_flux'])


# In[7]:


import matplotlib.pyplot as plt
plt.plot(ts.time.jd, ts['sap_flux'], 'k.', markersize=1)
plt.xlabel('julian date')
plt.ylabel('sap flux (e-/s)')
plt.show()


# In[36]:


import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares

periodogram = BoxLeastSquares.from_timeseries(ts, 'sap_flux')


# In[37]:


results = periodogram.autopower(0.2 * u.day, objective='snr')
best = np.argmax(results.power)
period = results.period[best]
print(period)


# In[38]:


print(results)


# In[39]:


plt.plot(results.period, results.power)
plt.show()


# In[40]:


transit_time = results.transit_time[best]
print(transit_time)


# In[41]:


print(transit_time.jd)


# In[42]:


ts_folded = ts.fold(period=period, epoch_time=transit_time)


# In[43]:


plt.plot(ts_folded.time.jd, ts_folded['sap_flux'], 'k.', markersize=1)
plt.xlabel('time')
plt.ylabel('sap flux (e-/s)')
plt.show()


# In[44]:


from astropy.timeseries import aggregate_downsample
from astropy.stats import sigma_clipped_stats


# In[45]:


mean, median, stddev = sigma_clipped_stats(ts_folded['sap_flux'])
ts_folded['sap_flux_norm'] = ts_folded['sap_flux'] / median
ts_binned = aggregate_downsample(ts_folded, time_bin_size=0.03 * u.day)
print(ts_binned)


# In[46]:


plt.plot(ts_folded.time.jd, ts_folded['sap_flux_norm'], 'k.', markersize=1)
plt.plot(ts_binned.time_bin_start.jd, ts_binned['sap_flux_norm'], 'b-', drawstyle='steps-post')
plt.xlabel('time')
plt.ylabel('normalized flux')
plt.show()


# In[48]:


max_power = np.argmax(results.power)
stats = periodogram.compute_stats(results.period[max_power],
                                  results.duration[max_power],
                                  results.transit_time[max_power])

print(stats)
# compute the max point on the periodogram


# # Lomb Scargle Periodograms

# In[49]:


rand = np.random.default_rng(42)


# In[50]:


t = 100 * rand.random(100)
y = np.sin(2 * np.pi * t) + 0.1 * rand.standard_normal(100)


# In[51]:


from astropy.timeseries import LombScargle


# In[52]:


frq, pwr = LombScargle(t, y).autopower()


# In[55]:


prd = np.argsort(t)
y = y[prd]
t = t[prd]


# In[56]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
axes[0].plot(frq, pwr)
axes[1].plot(t, y)
plt.show()


# In[59]:


best_frq = frq[np.argmax(pwr)]
t_fit = np.linspace(0, 1)
ls = LombScargle(t, y, 0.1 * (1 + rand.random(100)))
y_fit = ls.model(t_fit, best_frq)


# In[60]:


plt.plot(t_fit, y_fit)
plt.show()


# In[61]:


print(ls.false_alarm_probability(pwr.max()))


# In[62]:


# very insignifcant number = very good chance

