"""crop_coefficients.py
Defines CropCoeffs class
Defines functions read_crop_coefs_txt and read_crop_coefs_xls_xlrd to read
    crop coefficient data
Called by crop_et_data.py

"""

import numpy as np

"""
curve_descs : dict
    Crop curve type dictionary
    NCGDD (1)
    %PL-EC (2)
    %PL-EC+daysafter (3)
    %PL-Term (4)

"""

curve_descs = {'1': '1=NCGDD', '2': '2=%PL-EC', '3': '3=%PL-EC+daysafter', '4': '4=%PL-Term'}

class CropCoeff:
    """Crop coefficient container

    Attributes
    ----------
        curve_no :
            Crop curve number (1-60)
        curve_type_no :
            Crop curve type (1-4)
        curve_type :
            Crop curve type number
            (NCGDD, %PL-EC, %PL-EC+daysafter, %PL-Term)
        name : string
            Crop name
        data : ndarray
            Crop coefficient curve values

    Notes
    -----
    See comments in code

    """

    def __init__(self):
        """ """
        self.name = None
        self.gdd_type_name = ''

    def __str__(self):
        """ """
        return '<%s, type %s>' % (self.name, self.curve_types)

    def init_from_column(self, curve_no, curve_type_no, curve_name, data_col):
        """Parse column of data

        Parameters
        ----------
        curve_no : float

        curve_type_no :

        curve_name :

        data_col : string
            string of data column

        Returns
        -------
        None

        Notes
        -----
        See comments in code
        
        """

        self.curve_no = curve_no.replace('.0', '')
        self.curve_type_no = curve_type_no.replace('.0', '')
        self.curve_types = curve_descs[self.curve_type_no]
        self.name = curve_name

        # Data table
        values = data_col[0:35]    # this version's data_col already has header lines removed
        values = np.where(values == '', '0', values)
        self.data = values.astype(float)
        self.lentry = len(np.where(self.data > 0.0)[0]) - 1

def read_crop_coefs_txt(data):
    """Read crop coefficients from text file
    Parameters
    ---------
    data :

    Returns
    -------
    coeffs_dict : dict
        Dictionary of crop coefficients

    Notes
    -----
    See comments in code

    """

    a = np.loadtxt(data.crop_coefs_path, delimiter = data.crop_coefs_delimiter, dtype = 'str')
    curve_numbers = a[2, 2:]
    curve_type_numbs = a[3, 2:]    # repaired from 'a[2, 2:]' - dlk - 05/07/2016
    curve_names = a[4, 2:]
    coeffs_dict = {}
    for i, num in enumerate(curve_type_numbs):
        data_col = a[6:, 2 + i]
        if not curve_numbers[0]: continue
        coeff_obj = CropCoeff()
        coeff_obj.init_from_column(curve_numbers[i], curve_type_numbs[i], curve_names[i], data_col)
        coeffs_dict[int(coeff_obj.curve_no)] = coeff_obj
    return coeffs_dict

if __name__ == '__main__':
    pass
