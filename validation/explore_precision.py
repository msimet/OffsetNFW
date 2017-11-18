import numpy
try:
    import offset_nfw
except:
    import sys
    sys.path.append('..')
    import offset_nfw
import astropy.cosmology
import matplotlib.pyplot as plt
import astropy.units as u

def make_tables():
    cosmology = astropy.cosmology.Planck15
    model = offset_nfw.NFWModel(cosmology, precision=1E-3)
    mode.generate()
    # Explicitly deleting these to hopefully help with memory collection, since they're big
    del model
    model = offset_nfw.NFWModel(cosmology, precision=3E-3)
    model.generate()
    del model
    model = offset_nfw.NFWModel(cosmology, precision=1E-2)
    model.generate()
    del model
    model = offset_nfw.NFWModel(cosmology, precision=3E-2)
    model.generate()
    del model
    model = offset_nfw.NFWModel(cosmology, precision=1E-1)
    model.generate()
    print 3E-1
    model = offset_nfw.NFWModel(cosmology, precision=3E-1)
    model.generate()
    del model
    print 1
    model = offset_nfw.NFWModel(cosmology, precision=1)
    model.generate()
    del model
    print 3
    model = offset_nfw.NFWModel(cosmology, precision=3)
    model.generate()
    del model



def main():
    #TODO: make NFWModel kwargs configurable
    #TODO: redshifts?
    precision_list = [3E-2, 1E-1, 3E-1, 1, 3]
    numpy.random.seed(2012)
    cosmology = astropy.cosmology.Planck15
    model = offset_nfw.NFWModel(cosmology, deltasigma=False, sigma=False)
    x_range = model.x_range
    x_radii = numpy.exp(numpy.log10(x_range[0])+numpy.random.random(1000)*numpy.log10(x_range[1]))
    m, c = 1E14, 4
    
    z = 0.2
    radii = x_radii*model.scale_radius(m, c, z)
    ts = model.sigma_theory(radii, m, c, z)
    tds = model.deltasigma_theory(radii, m, c, z)
    max_ds_err = []
    max_s_err = []
    figsize = plt.gcf().get_size_inches()

    for j, precision in enumerate(precision_list):
        model = offset_nfw.NFWModel(cosmology, precision=precision)
        radii = x_radii*model.scale_radius(m, c, z).to(u.Mpc).value

        ds = model.deltasigma(radii, m, c, z)/tds
        plt.clf()
        plt.plot(radii, ds-1, 'ro')
        plt.xscale('log')
        plt.xlim((numpy.min(radii), numpy.max(radii)))
        plt.xlabel("$r$")
        plt.ylabel("$\widehat{\Delta\Sigma}/\Delta\Sigma$-1")
        plt.title("Fractional $\Delta\Sigma$ errors, precision=%.1E"%(
                        precision))
        plt.savefig("ds_frac_%.1e.png"%precision)
        plt.close(plt.gcf())
        max_ds_err.append(numpy.max(numpy.abs(1-ds)))
        
        s = model.sigma(radii, m, c, z)/ts
        plt.clf()
        plt.plot(radii, s-1, 'ro')
        plt.xscale('log')
        plt.xlim((min(radii), max(radii)))
        plt.xlabel("$r$")
        plt.ylabel("$\widehat{\Sigma}/\Sigma$-1")
        plt.title("Fractional $\Sigma$ errors, precision=%.1E"%(
                        precision))
        plt.savefig("s_frac_%.1e.png"%precision)
        plt.close(plt.gcf())
        max_s_err.append(numpy.max(numpy.abs(1-s)))
    
        miscentering_range = numpy.logspace(numpy.log10(x_range[0])+1, numpy.log10(x_range[1])-0.5, 
                                            num=100)*model.scale_radius(m, c, z)
        miscentering_range = miscentering_range.insert(0,0)
        rad_range = numpy.logspace(numpy.log10(x_range[0]), numpy.log10(x_range[1]), 
                                            num=120)*model.scale_radius(m, c, z).to(u.Mpc).value


        ds_arr = []
        s_arr = []
        for mr in miscentering_range:
            ds_arr.append(model.deltasigma(rad_range, m, c, z, mr))
            s_arr.append(model.sigma(rad_range, m, c, z, mr))
        max_ds = numpy.max(ds_arr)
        max_s = numpy.max(s_arr)
        
            
        plt.clf()
        for mr, ds in zip(miscentering_range, ds_arr)[::10]:
            p = plt.plot(rad_range, ds)
            color = p[0].get_color()
            plt.axvline(mr.to(u.Mpc).value, color=color)
        plt.xlabel("r")
        plt.xscale('log')
        plt.xlim((min(rad_range), max(rad_range)))
        plt.ylim((-0.1*max_ds, max_ds))
        plt.ylabel("$\Delta\Sigma$")
        plt.title("$\Delta\Sigma$, precision=%.1E"%(
                        precision))
        plt.savefig("miscentered_ds_%.1e.png"%precision)
        plt.close(plt.gcf())
            
        plt.clf()
        for mr, s in zip(miscentering_range, s_arr)[::10]:
            p = plt.plot(rad_range, s)
            color = p[0].get_color()
            plt.axvline(mr.to(u.Mpc).value, color=color)
        plt.xlabel("r")
        plt.xscale('log')
        plt.xlim((min(rad_range), max(rad_range)))
        plt.ylim((-0.1*max_s, max_s))
        plt.ylabel("$\Sigma$")
        plt.title("$\Sigma$, precision=%.1E"%(
                        precision))
        plt.savefig("miscentered_s_%.1e.png"%precision)
        plt.close(plt.gcf())

            
        ds_arr = numpy.array(ds_arr)
        s_arr = numpy.array(s_arr)
        plt.clf()
        fig = plt.figure(figsize=(2*figsize[0], 2*figsize[1]))
        plt.subplots(sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(25):
            axis = plt.subplot(5, 5, i+1)
            plt.plot(miscentering_range, ds_arr[:,4*i]/ds_arr[0,4*i])
#                plt.text(0.05, 0.05, "%.3f %s"%(miscentering_range[i].value, miscentering_range[i].unit),
#                         transform = axis.transAxes)
            if i%5!=0:
                plt.setp( plt.gca().get_yticklabels(), visible=False)
            if i<20:
                plt.setp( plt.gca().get_xticklabels(), visible=False)
            if i==10:
                plt.ylabel("$\Delta\Sigma$")
            if i==22:
                plt.xlabel("$r$")
        plt.savefig("miscentering_ds_smoothness_%.1e.png"%precision)
        plt.close(fig)
        
        plt.clf()
        fig = plt.figure(figsize=(2*figsize[0], 2*figsize[1]))
        plt.subplots(sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(25):
            axis = plt.subplot(5, 5, i+1)
            plt.plot(miscentering_range, s_arr[:,4*i]/s_arr[0,4*i])
#                plt.text(0.05, 0.05, "%.3f %s"%(miscentering_range[i].value, miscentering_range[i].unit),
#                         transform = axis.transAxes)
            if i%5!=0:
                plt.setp( plt.gca().get_yticklabels(), visible=False)
            if i<20:
                plt.setp( plt.gca().get_xticklabels(), visible=False)
            if i==10:
                plt.ylabel("$\Sigma$/")
            if i==22:
                plt.xlabel("$r$")
        plt.savefig("miscentering_s_smoothness_%.1e.png"%precision)
        plt.close(fig)
        
        # Miscentering against smallest choice
        # Repeat for rayleigh
        # Repeat for exponential
        # Test impact of angular binning

    plt.clf()
#    masses = [m for m, c in m_c_list]
    for mds, prec in zip(max_ds_err, precision_list):
        plt.plot(prec, mds)#, label='%.2f'%prec)
    plt.xlabel("Precision")
    plt.ylabel("Maximum fractional error on $\Delta\Sigma$")
#    plt.legend(loc='best')
    plt.savefig("max_ds_err.png")
    plt.close(plt.gcf())
        
    plt.clf()
    for ms, prec in zip(max_s_err, precision_list):
        plt.plot(prec, ms)#, label='%.2f'%prec)
    plt.xlabel("Precision")
    plt.ylabel("Maximum fractional error on $\Sigma$")
#    plt.legend(loc='best')
    plt.savefig("max_s_err.png")
    plt.close(plt.gcf())

if __name__=='__main__':
#    make_tables()
    main()
    
