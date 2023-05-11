# %%
import matplotlib.pyplot as plt # For ploting
import numpy as np # to work with numerical data efficiently

fs = 100 # sample rate 
f = 1.75 # the frequency of the signal

x = np.arange(fs) # the points on the x axis for plotting
# compute the value (amplitude) of the sin wave at the for each sample
y = np.sin(2*np.pi*f * (x/fs))

x = x[0:58]
y = y[0:58]

# showing the exact location of the smaples
plt.style.use('ggplot')

plt.figure(figsize=(15, 7))
plt.plot(x,y*-1, color='r')
plt.axvline(x = 57, color = 'b', linestyle='solid')
plt.text(-0.5, 0.1, "Disaster event", rotation='vertical', fontsize='x-large')
plt.text(58, 0.1, "New disaster event", rotation='vertical', fontsize='x-large')
plt.text(27, 0.1, "Business as usual", rotation='vertical', fontsize='x-large')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(2, -1.125, "Response and recovery", rotation='horizontal'
         , fontsize='x-large', bbox=props)
plt.text(38, 1.105, "PDRRPA", rotation='horizontal'
         , fontsize='x-large', bbox=props)
plt.axhline(y=0,linewidth=2, color='k', linestyle= 'dotted')
plt.ylim([-1.25, 1.25])

# Plot the second grid
fs = 100 # sample rate 
f = 1.75 # the frequency of the signal

x = np.arange(fs)+57 # the points on the x axis for plotting
# compute the value (amplitude) of the sin wave at the for each sample
y = np.sin(2*np.pi*f * (x/fs))

x = x[0:58]
y = y[0:58]

plt.plot(x,y*-1, color='r', label='Non-optimized helix')
plt.plot(x,y*-1*0.5, color='g', linestyle='dashed', linewidth=2, label='Optimized helix')


# Begin from PDRRPA

x = np.arange(fs)+29 # the points on the x axis for plotting
# compute the value (amplitude) of the sin wave at the for each sample
y = np.sin(2*np.pi*f * (x/fs))

x = x[0:28]
y = y[0:28]
plt.plot(x,y*-1*0.5, color='g', linestyle='dashed', linewidth=2)
plt.xticks([]) 
plt.yticks([]) 
plt.xlabel("Time (→)", fontsize='xx-large')
plt.ylabel("Resources/efforts (↔)", fontsize='xx-large')
plt.xlim([-2, 114])
plt.text(79, -1.200, "Adapted from Bosher et al. (2021)", rotation='horizontal'
         , fontsize='x-large')
plt.legend(loc="upper left")
plt.savefig('Helix.png', dpi=400, bbox_inches='tight')

# %%
