import numpy as np

# here are two examples of lines from the plot file.
#  Each line has 74 columns of data.
# the main problem comes in the third to last column where float values go over 100, and take up an extra space.
#  4894 4.155210267E+08   2.08600   3.69150   3.89125   4.00012   2.14250   0.00003   3.94196  -1.12865  -9.99568   3.99800  -3.99801   4.00009  -4.00009   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   1.62823   0.00000  -2.42322 4.995312E+01 7.308519E-01 2.634269E-01  4.75681E-03  2.02705E-04  5.19283E-04  0.00000E+00 -1.20327E+02  0.00000E+00  5.67111E+80  3.63144E+55  2.00000E+03  2.51792E+30  0.00000E+00  2.51792E+30     Infinity  0.00000E+00  1.93328E+02  9.34768E-07  0.00000E+00  1.52660E+00  1.72724  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  3.99812 99.82190  3.24903  8.04650
#  4895 4.155210776E+08   2.08817   3.69050   3.89157   4.00012   2.14249   0.00003   3.94216  -1.12124  -9.99568   3.99786  -3.99787   4.00009  -4.00009   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   1.62752   0.00000  -2.43380 5.088673E+01 7.308519E-01 2.634269E-01  4.75681E-03  2.02705E-04  5.19283E-04  0.00000E+00 -1.20322E+02  0.00000E+00  5.67111E+80  3.63144E+55  2.00000E+03  2.51792E+30  0.00000E+00  2.51792E+30     Infinity  0.00000E+00  1.97861E+02  9.27797E-07  0.00000E+00  1.52650E+00  1.72711  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  3.99800100.13230  3.24975  8.04666
# this is what I would like to avoid. using genfromtxt the 4th to last and 3rd to last columns are merged, causing an error.
#
# this code splits these values, and returns the full data array. 
#
ncols = 74  # expected number of columns

def read_plot_file_formatted(filename):
    data = []
    with open(filename, 'r') as f:
        for idx,line in enumerate(f):
            str_values = line.split()
            if len(str_values) == ncols:
                # correct number of columns, move on
                float_values = [float(val) for val in str_values]
                arr32 = np.array(float_values, dtype=np.float32)

                data.append(arr32)
                continue
            if len(str_values) == (ncols - 1):
                # split the 3rd to last column at the second decimal point
                # 3.99800100.13230  3.24975  8.04666
                # should then become
                # 3.99800  100.13230  3.24975  8.04666
                merged_value = str_values[-3]
                # split the merged value into two parts at the split index
                first_part = merged_value.split('.')[0]+'.'+merged_value.split('.')[1][:-3]
                second_part = merged_value.split('.')[1][-3:]+'.'+merged_value.split('.')[2]
                # replace the merged value with the two separate values
                str_values = str_values[:-3] + [first_part, second_part] + str_values[-2:]
                float_values = [float(val) for val in str_values]
                arr32 = np.array(float_values, dtype=np.float32)

                data.append(arr32)
            else:
                raise ValueError(f"Line has incorrect number of columns even after fixing: {line}")
    return np.array(data)