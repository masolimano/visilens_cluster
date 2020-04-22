import numpy as np
import copy
import visilens.visilens as vl
from skimage.transform import resize

def check_priors(p, source):
    """
    Checks if any of the proposed values in `p' lies outside
    the bounds given as priors.

    Parameters
    ---------
    p: array
        An array of the emcee-proposed steps for each of the
        free parameters
    source: list
        List of objects of
        any subclass of ~astropy.modeling.models.FittableModel2D

    Returns
    ------
    None if any value is outside the priors.
    Else return a copy of the passed objects with
    the free param values set to those in `p'.
    """
    source_copy = copy.deepcopy(source)

    ip = 0
    for i, src in enumerate(source):
        for param in src.param_names:
            if not src.fixed[param]:
                # Enforce bounds to be ordered
                minval, maxval = np.sort(src.bounds[param])

                if p[ip] < minval or p[ip] > maxval:
                    return None
                setattr(source_copy[i], param, p[ip])
                ip += 1
    # TODO: find a better way of incrementing the index for the
    # `ip` index

    return source_copy


def create_image(sources, xmap, ymap, dx=list(), dy=list()):
    """
    Produces an image of the source models in the source
    plane or the image plane.

    Parameters
    ---------
    sources : list
        List of objects of any subclass of
        ~astropy.modeling.models.FittableModel2D

    xmap : array
        Full resolution array of x-coordinates (+x is West).

    ymap : array
        Full resolution array of y-coordinates (+y is North).


    dx: list
        List of full-resolution x-deflection field arrays.
        Assumes each source has different redshift so
        it must have the same length and same indices as `sources`.
        If two or more source have the same redshift one could
        pass as many copies of the same deflection field
        as needed.
        If left empty will image the source-plane rather
        than the image-plane.

    dy: list
        List of full-resolution y-deflection field arrays.
        Assumes each source has different redshift so
        it must have the same length and same indices as `sources`.
        If two or more source have the same redshift one could
        pass as many copies of the same deflection field
        as needed.
        If left empty will image the source-plane rather
        than the image-plane.

    Returns
    ------
    field : array
        Raster image with shape xmap.shape (or ymap.shape, same shit)
    """
    # Init empty canvas for placing sources
    field = np.zeros_like(xmap) # could use ymap as well

    for i, model in enumerate(sources):
        model.bounding_box = None # Set to None because
        # otherwise will complain "aRraYs dO noT oVeRlAP".
        # It takes longer to run though, so I should
        # fix eventually

        if  len(dx) != 0 and len(dy) != 0:
            # Image the image-plane
            model.render(field, coords=(ymap - dy[i], xmap - dx[i]))
        else:
            # Image the source-plane
            model.render(field, coords=(ymap, xmap))

    return field

def log_likelihood(p, data, sources, ug, xmap, ymap):
    """
    Probability function of the parameters given
    the datasets and (uniform) priors.

    Parameters
    ---------
    p : array
        Array of proposed MCMC steps. Its length
        is the numbers of free parameters to fit.

    data : list
        List of ~visilens.VisData objects.

    sources : list
        List of objects of any subclass of
        ~astropy.modeling.models.Fittable2DModel

    ug : array
        Grid of visibilities to interpolate into.

    xmap : array
        Array of x-coordinates (+x is West).

    ymap : array
        Array of y-coordinates (+y is North).
    """
    source_list = check_priors(p, sources)
    if source_list is None:
        return -np.inf

    logL = 0.0
    for i, dset in enumerate(data):

        immap = create_image(source_list, xmap, ymap)
        interpdata = vl.fft_interpolate(dset, immap, xmap, ymap, ug)
        logL -= np.sum(np.hypot(dset.real - interpdata.real,
                                dset.imag - interpdata.imag) /\
                      dset.sigma ** 2)

    return logL

def log_likelihood_lens(p, data, sources, ug, xmap, ymap, lowx, lowy, dx, dy, npix=256):
    """
    Probability function of the parameters given
    the datasets and (uniform) priors.

    Parameters
    ---------
    p : array
        Array of proposed MCMC steps. Its length
        is the numbers of free parameters to fit.

    data : list
        List of ~visilens.VisData objects.

    sources : list
        List of objects of any subclass of
        ~astropy.modeling.models.Fittable2DModel

    ug : array
        Grid of visibilities to interpolate into,
        based on the low resolution images.

    xmap : array
        Full resolution array of x-coordinates (+x is West).

    lowx : array
        Low resolution array of x-coordinates (+x is West).

    lowy : array
        Low resolution array of y-coordinates (+y is North).

    ymap : array
        Full resolution array of y-coordinates (+y is North).

    dx: list
        List of full-resolution x-deflection field arrays.
        Assumes each source has different redshift so
        it must have the same length and same indices as `sources`.
        If two or more source have the same redshift one could
        pass as many copies of the same deflection field
        as needed.

    dy: list
        List of full-resolution y-deflection field arrays.
        Assumes each source has different redshift so
        it must have the same length and same indices as `sources`.
        If two or more source have the same redshift one could
        pass as many copies of the same deflection field
        as needed.

    npix: int
        Number of pixels per side of the low resolution images
    """
    source_list = check_priors(p, sources)
    if source_list is None:
        return -np.inf

    logL = 0.0
    for i, dset in enumerate(data):

        immap = create_image(source_list, xmap, ymap, dx, dy)
        immap = resize(immap, (npix, npix))

        interpdata = vl.fft_interpolate(dset, immap, lowx, lowy, ug)
        logL -= np.sum(np.hypot(dset.real - interpdata.real,
                                dset.imag - interpdata.imag) /\
                      dset.sigma ** 2)

    return logL


def init_ball(sources, nwalkers):
    """
    Creates an array of starting points for the walkers,
    which is a gaussian ball around the param values
    in the source models. The shape of the output
    array is (nwalkers, ndim), where ndim is the
    number of free parameters to fit.

    Parameters
    ---------
    sources: list
        List of objects of a subclass of ~astropy.models.FittableModel2D

    nwalkers: int
        Number of walkers of the emcee run.

    Returns
    -------
    ball: array
        (nwalkers, ndim)-shaped
        array of starting points for the walkers.

    """

    p  = list()
    scales = list()
    for i, src in enumerate(sources):
        for name in src.param_names:
            if not src.fixed[name]:
                p.append(getattr(src, name).value)
                diff = src.bounds[name][1] - src.bounds[name][0]
                scales.append(np.abs(diff)/6)
    p = np.array(p)
    scales = np.array(scales)

    # p.size is ndim, the number of parameters to fit
    # here we adjust the scale (sigma) of the gaussian
    # to be one sixth of the prior range length
    ball = p + scales * np.random.randn(nwalkers, p.size)
    return ball
