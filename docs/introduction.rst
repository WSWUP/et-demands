Introduction
============
The ET-Demands model develops daily estimates of crop irrigation water requirements
using daily weather data (including reference evapotranspiration) along with
crop-specific crop growth curves.  

Acknowledgements
----------------
The basis for the ET Demands model is the dual crop coefficient method
presented in FAO-56 `(Allen et al., 1998) <https://www.kimberly.uidaho.edu/water/fao56/fao56.pdf>`_.
Original model code was developed by Dr. Rick Alan at the University of Idaho
Kimberly Reserarch and Extension Center in Kimberly, ID.

Model History
-------------
Original model code was written in Visual Basic for Windows. Reclamation and
the Desert Research Institute ported this code to a non-versioned Python release
of the model. This non-versioned Python release served as the basis for the
et-demands model found in this repository.
This methods used in this model were used to evaluate evapotranspiration and
consumptive use irrigation water requirements for the state of Idaho
`(Allen and Robison, 2007) <http://data.kimberly.uidaho.edu/ETIdaho/ETIdaho_Report_April_2007_with_supplement.pdf>`_
and the state of Nevada `(Huntington and Allen, 2010) <https://www.dri.edu/images/stories/divisions/dhs/dhsfaculty/Justin-Huntington/Huntington_and_Allen_2010.pdf>`_.
This approach has been used to quantify historical and future irrigation water
requirements for selected irrigation projects operated by the Bureau of
Reclamation `(Reclamation, 2016) <https://www.usbr.gov/watersmart/baseline/docs/historicalandfutureirrigationwaterrequirements.pdf>`_.
This approach has also been used to quantify historical and future irrigation
water requirements in support of Reclamation's
WaterSMART `Basin Studies Program <https://www.usbr.gov/watersmart/bsp/>`_.

Model Version History
---------------------
v1.0.0
^^^^^^
**Release date: July 15, 2019**

This is the first official release of the et-demands model containing
workable source code, examples, and documentation. The dual crop coefficient
curve approach remains unchanged from previous releases. This version has undergone
thorough testing to ensure reliability, with pre-processing and post-processing tools
to develop the required input data, and format and visualize model results are much
improved. Model workflow has been better refined to rely on ESRI Shapefiles and
daily weather timeseries data as the main

v0.1.0
^^^^^^
**Release date: September 20, 2018**

This is a beta release of the et-demands model containing workable source code.
The dual crop coefficient curve approach remains unchanged from the original
non-versioned Python code, developed from the Visual Basic version of the et-demands model.
Structural changes to the code were made to make ESRI Shapefiles and daily weather
timeseries the primary source data for running the model. Pre-processing and
post-processing tools were written to develop the required input data, and format
and visualize model result using the ESRI Shapefile and daily weather timeseries
within a standard workflow. Further refinement and testing of this workflow,
development of workable examples and source code are required for a v1.0.0
release.

Model License
-------------

The software as originally published constitutes a work of the United States
Government and is not subject to domestic copyright protection under 17 U.S.C.
§ 105. Subsequent contributions by members of the public, however, retain
their original copyright.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
