import numpy
import astropy.cosmology
import os, glob
import time
import matplotlib.pyplot as plt

try:
    import offset_nfw
except:
    import sys
    sys.path.append('..')
    import offset_nfw
    
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
plt.subplots(sharex=True, sharey=True)
fontsize = 20

def set_plt_font(s):
    rcParams['axes.labelsize'] = s
    rcParams['xtick.labelsize'] = s
    rcParams['ytick.labelsize'] = s
    rcParams['legend.fontsize'] = s
set_plt_font(fontsize)

yellow = '#F0E442'
blue = '#0072B2'
vermilion = '#D55E00'
teal = '#009E73'


def plot_three(x, yref, ytest, yref_kwargs, ytest_kwargs, ylabel, title='', slug=''): 
    plt.clf()
    plt.subplot(311)
    plt.plot(rad_bins, yref, **yref_kwargs)
    plt.plot(rad_bins, ytest, **ytest_kwargs)
    plt.ylabel(ylabel+' (linear)')
    plt.xscale('log')
    plt.setp( ax.get_xticklabels(), visible=False)
    plt.legend(loc='best')

    plt.subplot(312)
    plt.plot(rad_bins, yref, **yref_kwargs)
    plt.plot(rad_bins, ytest, **ytest_kwargs)
    plt.yscale('log')
    plt.ylabel(ylabel+' (log)')
    plt.xscale('log')
    plt.setp( ax.get_xticklabels(), visible=False)
    
    plt.subplot(313)
    plt.plot(rad_bins, ytest/yref-1, **ytest_kwargs)
    plt.ylabel('$\Delta$ '+ylabel)
    plt.xlabel("$R$ [Mpc]")
    plt.yscale('log')
    plt.xscale('log')
    
    plt.title(title)
    
    plt.savefig('test_validation_%s.png'%slug)
    

def pregenerate():
    cosmo = astropy.cosmology.FlatLambdaCDM(H0=100, Om0=0.3)
    halo_1em3 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c', precision=3E-3)
#    if not hasattr(halo_1em3, '_sigma'):
    halo_1em3.generate()

def main():
    M, c, z = 1E14, 4, 0.2
    time.sleep(20)
    # Build an object to play with
    cosmo = astropy.cosmology.FlatLambdaCDM(H0=100, Om0=0.3)
    halo = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')    
    
    # Get rid of existing files
    existing_files = glob.glob(halo.table_file_root+'*.npy')
    for ef in existing_files:
        os.remove(ef)
    halo = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')    
    
    # Build reference halos
#    halo_1em3 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c', precision=3E-3)
#    if not hasattr(halo_1em3, '_sigma'):
#        halo_1em3.generate()
#    halo_1em3 = None

    rad_edges = halo.scale_radius(M, c, z).value*numpy.array(halo.x_range)
#    rad_bins = numpy.logspace(numpy.log10(rad_edges[0]), numpy.log10(rad_edges[1]), num=55000)

    # Now, the default precision goes like this:
    # nxbins = int(25/precision)
    # nthetabins = int(2pi/precision)
    # so for precision == 0.01, which we WANT to be one percent precision, 
    # nxbins = 2500
    # nthetabins = 314
    
    theory_kwargs = {'linestyle': 'dashed', 'color': blue, 'lw': 3, 'label': 'Theory'}
    ref_kwargs = {'linestyle': 'dotted', 'color': teal, 'lw': 3, 'label': 'Reference table'}
    test_kwargs = {'linestyle': 'solid', 'color': vermilion, 'lw': 3, 'label': 'Test precision'}
    
    
    for nxbins, nthetabins in [(2500, 314)]:
        halo.generate(nxbins=nxbins, nthetabins=nthetabins)
        halo_1em3 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c', precision=3E-3)
        ds_theory = halo.deltasigma_theory(rad_bins, M, c, z)
        ds_interp = halo.deltasigma(rad_bins, M, c, z)
        plot_three(rad_bins, ds_theory, ds_interp, theory_kwargs, test_kwargs, 
                   '$\Delta\Sigma$', 'nxbins=$i, nthetabins=$i'%(nxbins, nthetabins), 
                   'deltasigma_%i_%i')
        
        
    
    
    
    
if __name__=='__main__':
    pregenerate()
#    main()    
