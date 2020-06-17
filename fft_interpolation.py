from visilens.visilens import Visdata
import numpy as np
from scipy.interpolate import RectBivariateSpline

def fft_interpolate2(visdata, immap, xmap, ymap, uvgrid=None, scaleamp=1., shiftphase=[0., 0.]):
      """
      Take a dataset and a map of a field, fft the image,
      and interpolate onto the uv-coordinates of the dataset.
      Returns:
      interpdata: Visdata object
            A dataset which samples the given immap
      """

      # Correct for PB attenuation
      if visdata.PBfwhm is not None:
            PBs = visdata.PBfwhm / (2 * np.sqrt(2 * np.log(2)))
            immap *= np.exp(-(xmap ** 2 / (2 * PBs ** 2)) - (ymap ** 2 / (2 * PBs ** 2)))

      #immap = immap[::-1,:] # Fixes issue of origin in tlc vs blc to match sky coords
      imfft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(immap)))

      # Calculate the uv points we need, if we don't already have them
      if uvgrid is None:
            xkmax = 0.5 / (np.abs(xmap[0, 1] - xmap[0, 0]) * arcsec2rad)
            ykmax = 0.5 / (np.abs(ymap[0, 1] - ymap[0, 0]) * arcsec2rad)
            ug = np.linspace(-xkmax, xkmax, xmap.shape[0])
            vg = np.linspace(-ykmax, ykmax, ymap.shape[1])

      else:
          ug, vg = uvgrid

      # Interpolate the FFT'd image onto the data's uv points
      # Using RBS, much faster since ug is gridded
      spliner = RectBivariateSpline(vg, ug, imfft.real, kx=1, ky=1)
      splinei = RectBivariateSpline(vg, ug, imfft.imag, kx=1, ky=1)
      interpr = spliner.ev(visdata.v, visdata.u)
      interpi = splinei.ev(visdata.v, visdata.u)
      interpdata = Visdata(visdata.u, visdata.v, interpr,interpi, visdata.sigma,\
            visdata.ant1, visdata.ant2, visdata.PBfwhm,'interpolated_data')

      # Apply scaling, phase shifts; wrap phases to +/- pi.
      interpdata.amp *= scaleamp
      interpdata.phase += 2 * np.pi * arcsec2rad * (shiftphase[0] * interpdata.u + shiftphase[1] * interpdata.v)
      interpdata.phase = (interpdata.phase + np.pi) % (2 * np.pi) - np.pi

      return interpdata
