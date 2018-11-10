This project compresses and reconstructs infrared hyperspectrum data. The network proposed is named HCR, aka Hyper-spectrum compression and reconstruction. The size of infrared hyperspectrum data are so big that they are overload for current computing resources. Taking IASI, an atmosphere detector on satellite Metop launched by European Organization for the Exploitation of Meteorological Satellites (EUMETSAT), as an example, it has 8461 channels, which can detect atmosphere vertically in details. To process these data more efficiently, compressing them and then reconstructing is required.

Considering their high correlation in spectral and spatial dimension, a new compressing and reconstructing network HCR is proposed. Concretely, the radiation brightness value are gridded so that one value at specific location is taken as a color value at this pixel. After normalizing with batch normalization, HCR compresses by convolution and reconstructs by deconvlution.

Carrying on IASI data, the RMSE of this new method was 5 smaller than that of principle component analysis (PCA) at least in the same compression ratio. The compression kernels code tempetature information and reconstruct it. During reconstruction, the kernels' weights for likewise data are similar.

#  Experiments
1.  EX1: Verify the efficiency of proposed HCR.
    Datasets are gridded, and each grid covers 10 degree both in latitude and longitude. The resolution of data is set to 0.1. Both PCA and HCR are used to show the improvement after urilizing HCR.
2.  Ex2: Visualize the kernels weights of the first layer while compressing and the last layer while reconstructing to analyze what information are coded and reconstructed.
 
# Codes' Framework 
----data		: The experiment data.

----extract		: Grid data.

----model		: HCR and PCA.

----estimate	: Analyze results.

----util		: Utilities for IO, plotting, etc.

# Encoding
The files in this project is encoded in UTF-8, either English characters or Chinese characters.