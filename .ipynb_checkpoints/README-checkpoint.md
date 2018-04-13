# What the Flux?
[1.1]: http://i.imgur.com/tXSoThF.png
[1]: http://www.twitter.com/cfrohmaier

[![alt text][1.1]][1] [@cfrohmaier](http://www.twitter.com/cfrohmaier "@cfrohmaier") 

[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

This repository was designed to help astronomers with their unit conversions.

There are examples of what goes on behind the scenes in the various jupyter notebooks that I will add to over time.


## Intro to Magnitudes

The magnitude system is based on a naked eye visual assessment by the Greek astronomer Hipparchos. The brightest stars he could see were assigned a magnitude of 1 and the faintest, a magnitude of 6. However, our eyes do not respond linearly to light and therefore a 6th magnitide object is not 6x fainter than a 1st magnitude star. We have now formalized the scale definition of magnitudes, so that 6th magnitude objects are 100x fainter than first magnitude objects. Therefore, each step in the magnitude system a brightness change of 100^{1/5} = 2.512.

The ratio of brightnesses can be stated as: F1 / F2 = 2.512^(-(m1 -m2))

or: log10(F1/F2) = -(m1 - m2) * log10(2.512) = -0.4*(m1 - m2)

or: m1 - m2 = -2.5 log10 (F1/F2)

Where F represents the fluxes of our stars and m the magnitudes. Reminder that fainter stars have a larger magnitude, hence when the exponent is raised to the negative of the difference. **Note:** I have **not** rounded -2.512 to -2.5, be aware that log10(2.512) = 1/2.5, wow!

The magnitudes system is a **relative brightness** 

## Fluxes

Please refer to the jupyter notebook [`Fluxes_and_AB.ipynb`](https://github.com/chrisfrohmaier/what_the_flux/blob/master/Fluxes_and_AB.ipynb)  for more in depth discussion on flux systems and some demonstration code.