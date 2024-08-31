
#### The data-processing codes for alpha-LFM
***
Directory Structure:

```    
├── utils
    └── misc: 
        Functions for image I/O, multi-views extraction for LF images, etc.
    └── SASProjection:
        Functions for converting 3D images to synthetic sub-aperture shifted LF. To use this, a PSF file is required. We use softwares provided in 
        Prevedel, R., Yoon, Y., Hoffmann, M. et al. Simultaneous whole-animal 3D imaging of neuronal activity using light-field microscopy. Nat Methods 11, 727–730 (2014).https://doi.org/10.1038/nmeth.2964 
        to compute Light Field PSF.
    └── jsonlab 2.0: 
        The third-party codes to load/save ".json" file
        Copyright (c) 2011-2024 Qianqian Fang <q.fang at neu.edu>
        License: BSD or GNU General Public License version 3 (GPL v3)
        Version: 2.9.8 (Micronus Prime - Beta)
        URL: https://neurojson.org/jsonlab
```