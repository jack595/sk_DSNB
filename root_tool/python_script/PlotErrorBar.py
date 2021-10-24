import numpy as np
import matplotlib.pylab as plt
import histlite as hl
def PlotErrorBandWithBinEdge( bin_edge, error_band_lower, error_band_upper, ax:plt.Axes):
    x_return = []
    y_upper_return = []
    y_lower_return = []
    for i in range(len(bin_edge)-1):
        x_return.append(bin_edge[i])
        y_upper_return.append(error_band_upper[i])
        y_lower_return.append(error_band_lower[i])
        x_return.append(bin_edge[i+1])
        y_upper_return.append(error_band_upper[i])
        y_lower_return.append(error_band_lower[i])
    ax.fill_between(x_return, y_upper_return , y_lower_return,alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=4, linestyle='dashdot', antialiased=True)
def PLotDataWithErrorBar(h_data:hl.Hist, ax_plot:plt.Axes):
    bins_width=np.diff(h_data.bins)[0]/2
    bins_h_data=np.array(h_data. bins[0])
    bins_center=(bins_h_data[:-1]+bins_h_data[1:])/2
    bins_error=np. sqrt(h_data. values)/2
    ax_plot.errorbar(bins_center,h_data.values, xerr=bins_width, yerr=bins_error, marker="+",color="black", Ls="none", Label="Input Events")
def PLotDataWithErrorBar_numpy(h_data:np.ndarray, h_edges:np.ndarray,h_y_errors:np.ndarray=None, label="",
                               color=None, density=False):
    if np.all(h_y_errors) == None:
        h_y_errors = h_data**0.5
    if density:
        h_y_errors = h_y_errors/np.sum(h_data)
        h_data = h_data/np.sum(h_data)

    bins_width=np.diff(h_edges)/2
    bins_center=(h_edges[:-1]+h_edges[1:])/2
    bins_error= h_y_errors
    plt.errorbar(bins_center,h_data, xerr=bins_width, yerr=bins_error, marker="+",markersize=5,color=color, Ls="none", Label=label)

def PlotRatioOfTwoSpectrum(h1_data:np.ndarray, h2_data:np.ndarray, h_edges:np.ndarray,
                           h1_yerr:np.ndarray=None, h2_yerr:np.ndarray=None):
    if np.all(h1_yerr) ==None:
        h1_yerr = h1_data**0.5
    if np.all(h2_yerr) ==None:
        h2_yerr = h2_data**0.5

    # Normalize two spectrum
    h1_data_norm = h1_data/np.max(h1_data)
    h2_data_norm = h2_data/np.max(h2_data)
    h1_yerr = h1_yerr/np.max(h1_data)
    h2_yerr = h2_yerr/np.max(h2_data)

    sigma = ( (h1_yerr/h1_data_norm)**0.5+(h2_yerr/h2_data_norm)**0.5 )**0.5
    ratio = h1_data_norm/h2_data_norm
    PLotDataWithErrorBar_numpy(ratio, h_edges, sigma)







