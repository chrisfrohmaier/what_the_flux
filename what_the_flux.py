import os
import numpy as np
import scipy.interpolate as spint
import astropy.units as u
import astropy.constants as const
import scipy.ndimage as ndimage
import extinction
from astropy.io import fits
from functools import cached_property

if hasattr(np, 'trapezoid'):
    trapz = np.trapezoid
else:
    trapz = np.trapz


def get_color_band(band_name):
    color_band = {'A':'grey',
            'U': 'blue',
            'B': 'royalblue',
            'V':  'limegreen',
            'R':  'red',
            'I':  'mediumvioletred',
            'g':    'darkgreen',
            'i':    'purple',
            'r':    'darkred',
            'z':    'sienna',
            'u': 'darkblue',
            'UVW1':'purple',
            'UVW2':'pink',
            'UVM2':'m',
            'H':'brown',
            'J':'salmon',
            'K':'sienna'}
    if band_name in color_band.keys():
        return color_band[band_name]
    else:
        return 'grey'

def loadFilter(dir, natural=True):
    '''In this function, the files cotaining the filter
    transmission functions are loaded and returns the wavelengths and
    transmission functions

    NATURAL = TRUE means 'This filter has NOT been multiplied for lambda. So what im reading from the file is just the transmission of the filter'
    NATURAL = FALSE means 'This filter has already been multiplied for lambda. So what im reading from the file is not just the transmission of the filter itself but transmission*lambda'
    '''
    w, t = np.loadtxt(dir,usecols=(0,1), dtype=np.float64, unpack=True)

    if max(t)>1.:
        t = t/100.

    if natural: return w, (t)/max(t)
    else: return w, (t/w)/max(t/w)

def fluxLamFilter(wave, filt_wave, fil_trans, F_lam):
    w_val = wave.value if hasattr(wave, 'value') else wave
    fw_val = filt_wave.value if hasattr(filt_wave, 'value') else filt_wave
    
    filt_int = np.interp(w_val, fw_val, fil_trans, left=0, right=0)
    filtspec = filt_int*F_lam
    fluxFilt = trapz(filtspec, x=wave)
    return fluxFilt

def countFilter(wave, filt_wave, fil_trans, F_cnt):
    w_val = wave.value if hasattr(wave, 'value') else wave
    fw_val = filt_wave.value if hasattr(filt_wave, 'value') else filt_wave
    
    filt_int = np.interp(w_val, fw_val, fil_trans, left=0, right=0)
    filtspec = filt_int*F_cnt
    countFilt = trapz(filtspec, x=wave)
    if isinstance(countFilt, u.quantity.Quantity):
        return countFilt.value
    else: return float(countFilt)

def mag2flux(band,magnitude, perArea=True):
    return band.zpFlux(perArea=perArea)*10.**(magnitude/-2.5)

def ERRmag2ERRflux(band, magnitude, err_magnitude, perArea=True):
    #  fZP * 10**(-0.4x)* ln(10) * (-0.4) *err_mag
    return band.zpFlux(perArea=perArea)* np.abs(10**(-0.4*magnitude) * np.log(10.)*(-0.4)*err_magnitude)

def flux2mag(band,flux, perArea=True):
    return -2.5*np.log10(flux.value / band.zpFlux(perArea=perArea).value)

def ERRflux2ERRmag(band,flux, err_flux, perArea=True):
    return 2.5/np.log(10.) * err_flux/flux

def mag2Counts(band, magnitude,totalThrough=True):
    if totalThrough==True:
        return band.zpCnt(perArea=True)*10.**(magnitude/-2.5) * band.areaA()
    else:
        return band.zpCnt(perArea=False)*10.**(magnitude/-2.5)


class BaseBand(object):
    def __init__(self, wave, transmission):
        if wave.shape != transmission.shape:
            raise ValueError('The wave array and transmission array are different sizes!')
        self.wave = wave * u.AA
        self.transmission = transmission
        self._areaA = np.abs(trapz(self.transmission, x=self.wave))

    def filterFLambdaBins(self):
        return np.arange(min(self.wave.value), max(self.wave.value)+1., 1.) * u.AA

    def areaA(self):
        return self._areaA

    def zpAngstrom(self, perArea=True):
        zp_dum = self._zpFluxesFlam()
        flux = fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])
        if perArea:
            return -2.5*np.log10(flux.value / self.areaA().value)
        else:
            return -2.5*np.log10(flux.value)

    def zpFlux(self, perArea=True):
        zp_dum = self._zpFluxesFlam()
        flux = fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])
        if perArea:
            return flux / self.areaA().value
        else:
            return flux

    def zpFluxFnu(self, perArea=True):
        F0lam = self.zpFlux(perArea=perArea)
        return (F0lam / const.c.cgs * ((self.effWave())**2)).to(u.erg/u.s/u.cm**2/u.Hz)

    def zpCnt(self, perArea=True):
        zp_dum = self._zpFluxesFcnt()
        cnt = countFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])
        if perArea:
            return cnt / self.areaA().value
        else:
            return cnt

    def meanWave(self):
        return np.average(self.wave, weights=self.transmission)

    def effWave(self):
        fw = np.interp(self.wave.value, self.filterFLambdaBins().value, self._zpFluxesFlam()[1].value)
        n1 = self.wave * self.transmission * fw
        d1 = self.transmission * fw
        return trapz(n1, x=self.wave) / trapz(d1, x=self.wave)

    def pivot(self):
        lt = trapz(self.wave * self.transmission, x=self.wave)
        return np.sqrt(lt / trapz(self.transmission / self.wave, x=self.wave))

    def peakWave(self):
        return self.wave[self.transmission == max(self.transmission)][0]

    def minWave(self):
        return min(self.wave)

    def maxWave(self):
        return max(self.wave)

    def wiredfunc(self):
        int1 = trapz(self.transmission / self.wave, x=self.wave)
        int2 = trapz(self.transmission / (self.wave)**2, x=self.wave)
        int3 = trapz(self.transmission, x=self.wave)
        return int1**2 / (int2 * int3)

    def extinction(self, eb_v, dust_law, r_v=3.1):
        a_v = eb_v * r_v
        w = self.wave.value
        t = self.transmission
        
        if dust_law == 'OD94':
            a_lambda = extinction.od94(w, a_v, r_v)
        elif dust_law == 'CCM':
            a_lambda = extinction.ccm89(w, a_v, r_v)
        elif dust_law == 'F99':
            a_lambda = extinction.fitzpatrick99(w, a_v, r_v)
        else: 
            print('Add this dust law! I dont know it')
            return None

        ext = t * 10**(-0.4 * a_lambda)
        correction_extinction = trapz(ext, x=w) / trapz(t, x=w)
        return correction_extinction

    def _zpFluxesFlam(self):
        raise NotImplementedError

    def _zpFluxesFcnt(self):
        raise NotImplementedError

    def _zpFluxesFnu(self):
        raise NotImplementedError


class Band_AB(BaseBand):
    """The transmission function of a filter in AB.
    The transmission is a fraction of 1.
    The wavelengths are in Angstrom
    """
    def _Fnu(self):
        return 3631e-23 * (u.erg/u.s/u.cm**2/u.Hz)

    @cached_property
    def _cached_zpFluxesFlam(self):
        wavelength_array = self.filterFLambdaBins()
        F_nu = self._Fnu()
        F_lam = (F_nu * const.c.cgs / (wavelength_array**2)).to(u.erg/u.s/u.cm**2/u.AA)
        return wavelength_array, F_lam

    def _zpFluxesFlam(self):
        return self._cached_zpFluxesFlam

    @cached_property
    def _cached_zpFluxesFcnt(self):
        wavelength_array = self.filterFLambdaBins()
        F_nu = self._Fnu()
        F_lam = (F_nu * const.c.cgs / (wavelength_array**2)).to(u.erg/u.s/u.cm**2/u.AA)
        F_cnt = ((F_lam.value * wavelength_array.value) / (const.c.cgs.value * const.h.cgs.value)) * 1e-8
        return wavelength_array, F_cnt

    def _zpFluxesFcnt(self):
        return self._cached_zpFluxesFcnt

    def _zpFluxesFnu(self):
        return self.filterFLambdaBins(), self._Fnu() * np.ones(len(self.filterFLambdaBins()))


class Band_Vega(BaseBand):
    """The transmission function of a filter in Vega.
    The transmission is a fraction of 1.
    The wavelengths are in Angstrom
    """
    @cached_property
    def vega_spectrum(self):
        module_dir = os.path.dirname(__file__)
        vega_path = os.path.join(module_dir, 'alpha_lyr_stis_006.fits')
        with fits.open(vega_path) as vega:
            w_Vega_full = vega[1].data['WAVELENGTH']
            f_Vega_full = vega[1].data['FLUX']
            mask_Vega = (w_Vega_full > 500.) & (w_Vega_full < 25000)
            return w_Vega_full[mask_Vega], f_Vega_full[mask_Vega]

    @cached_property
    def _cached_zpFluxesFlam(self):
        wavelength_array = self.filterFLambdaBins()
        w_Vega, f_Vega = self.vega_spectrum
        F_Vega_interp = np.interp(wavelength_array.value, w_Vega, f_Vega) * (u.erg/u.s/u.cm**2/u.AA)
        return wavelength_array, F_Vega_interp

    def _zpFluxesFlam(self):
        return self._cached_zpFluxesFlam

    @cached_property
    def _cached_zpFluxesFcnt(self):
        wavelength_array, F_lam = self._zpFluxesFlam()
        F_cnt = ((F_lam.value * wavelength_array.value) / (const.c.cgs.value * const.h.cgs.value)) * 1e-8
        return wavelength_array, F_cnt

    def _zpFluxesFcnt(self):
        return self._cached_zpFluxesFcnt

    @cached_property
    def _cached_zpFluxesFnu(self):
        wavelength_array, F_lam = self._zpFluxesFlam()
        F_nu = (F_lam / const.c.cgs * (wavelength_array**2)).to(u.erg/u.s/u.cm**2/u.Hz)
        return wavelength_array, F_nu

    def _zpFluxesFnu(self):
        return self._cached_zpFluxesFnu


class Spectrum(object):
    """docstring for Spectrum."""
    def __init__(self,wave,flux):
        super(Spectrum, self).__init__()
        self.wave = wave
        self.flux = flux

    def bandflux(self, band):
        """Do some integration through a filter in this function
        """
        w_val = self.wave.value if hasattr(self.wave, 'value') else self.wave
        bw_val = band.wave.value if hasattr(band.wave, 'value') else band.wave
        filt_int = np.interp(w_val, bw_val, band.transmission)
        filtspec = filt_int*self.flux
        flux1 = trapz(filtspec, x=self.wave)/band.areaA()
        return flux1

    def runningSmooth(self, AAbin=None, NumBin=None):
        if AAbin is None and NumBin is None:
            raise ValueError('You must provide a binning. Either AAbin to provide the angstrom bins (rounded to nearest integer) or NumBin for number of bins')
        if NumBin is not None and not float(NumBin).is_integer():
            raise ValueError('NumBin must be an integer')

        if AAbin is not None:
            medDiff = np.median(np.diff(self.wave))
            nbins = int(AAbin/medDiff)
        else:
            nbins = int(NumBin)
            
        self.wave = np.convolve(self.wave, np.ones((nbins,))/nbins, mode='same')
        self.flux = np.convolve(self.flux, np.ones((nbins,))/nbins, mode='same')

    def binSpec(self, nbins, median=True):
        """median or mean bin the spectrum
        """
        if median:
            self.wave = ndimage.median_filter(self.wave, size=nbins, mode='reflect')
            self.flux = ndimage.median_filter(self.flux, size=nbins, mode='reflect')
        else:
            self.wave = ndimage.uniform_filter(self.wave, size=nbins, mode='reflect')
            self.flux = ndimage.uniform_filter(self.flux, size=nbins, mode='reflect')

    def filterOverlap(self, band):
        """This function checks to see if there is complete, partial, or no overlap of the spectrum
        with the desired band"""
        low1 = min(band.wave[band.transmission>1e-3])
        hi1 = max(band.wave[band.transmission>1e-3])

        low2 = min(self.wave)
        hi2 = max(self.wave)
        inarr = low1>low2 and hi1<hi2
        return inarr
