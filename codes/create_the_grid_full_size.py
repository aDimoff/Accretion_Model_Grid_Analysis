print('Creating Massive Model Array for Accretion Model Grid')
print()
print('collecting packages...')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.io import fits, ascii
import pickle
from PyAstronomy import pyasl
from time import sleep
from tqdm.notebook import tqdm
import math

from formatted_read_plot_file import read_plot_file_formatted

import warnings
warnings.filterwarnings("ignore")

# convert model values for mass [6] and radius [3] into cgs values
# log (G M / R^2)
bigG = 6.67430e-8
solar_mass_grams = 1.98847e33
solar_radius_cm = 6.95700e10

print('Reading list of files...')
data_dir = '/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/' #'Accretion_New/acc_z0001/'
list_plot_files = np.genfromtxt(data_dir+'list_plot_files_full',dtype=str)
list_surf_files = np.genfromtxt(data_dir+'list_surface_files_full',dtype=str)
print(len(list_plot_files),'models in the grid')

if (len(list_plot_files) != len(list_surf_files)):
    print(' Number of plot files does not equal the number of surface files ')
    print(' Please check your input files! ')
    
# sum of all isotopes of elements from Yield Conversion Notebook
# Fe, Sr, Y, Zr, (Nb), Mo, (Ru), Ba, La, Ce, [Pr], Nd, (Sm), Eu, [Dy], Pb
# AGB yields from FRUITY at z = 0.0001
heavy_yields_m1p3_z0001 = np.array([9.120538E-06 ,   4.686225E-09 ,   8.033977E-10 ,   1.534570E-09 ,   9.728278E-11 ,   2.522094E-10 ,   1.457713E-10 ,   2.110597E-09 ,   2.139004E-10 ,   7.031063E-10 ,   7.074584E-11 ,   4.205315E-10 ,   8.235655E-11 ,   8.393006E-12 ,   7.334964E-11 ,   6.637583E-08])
heavy_yields_m1p5_z0001 = np.array([9.111998E-06 ,   3.210986E-09 ,   7.647503E-10 ,   1.911643E-09 ,   1.318462E-10 ,   3.357407E-10 ,   1.856965E-10 ,   3.524221E-09 ,   3.638761E-10 ,   1.227137E-09 ,   1.269636E-10 ,   7.130932E-10 ,   1.470578E-10 ,   1.328370E-11 ,   1.243211E-10 ,   1.038645E-07])
heavy_yields_m2p0_z0001 = np.array([9.067902E-06 ,   3.565077E-09 ,   9.792717E-10 ,   2.569341E-09 ,   1.844625E-10 ,   4.747969E-10 ,   2.346975E-10 ,   4.340988E-09 ,   4.609777E-10 ,   1.540274E-09 ,   1.626370E-10 ,   9.033580E-10 ,   2.034491E-10 ,   1.782235E-11 ,   1.896710E-10 ,   1.171073E-07])
heavy_yields_m2p5_z0001 = np.array([8.981697E-06 ,   5.739964E-09 ,   1.445469E-09 ,   3.655464E-09 ,   2.392478E-10 ,   7.270598E-10 ,   4.145074E-10 ,   4.628509E-09 ,   4.441695E-10 ,   1.272120E-09 ,   1.255825E-10 ,   9.161112E-10 ,   2.400566E-10 ,   2.097872E-11 ,   2.250907E-10 ,   1.058656E-07])
heavy_yields_m3p0_z0001 = np.array([9.031263E-06 ,   7.461499E-09 ,   1.784572E-09 ,   4.264786E-09 ,   2.843673E-10 ,   8.412536E-10 ,   4.837237E-10 ,   4.040576E-09 ,   3.280334E-10 ,   8.067842E-10 ,   8.096076E-11 ,   5.628014E-10 ,   1.474214E-10 ,   1.392974E-11 ,   1.412262E-10 ,   6.129123E-08])
heavy_yields_m4p0_z0001 = np.array([9.029437E-06 ,   1.139257E-08 ,   2.656978E-09 ,   5.841225E-09 ,   3.959204E-10 ,   1.133086E-09 ,   6.512108E-10 ,   4.219564E-09 ,   2.574852E-10 ,   4.745300E-10 ,   5.080968E-11 ,   3.138580E-10 ,   8.059500E-11 ,   8.727057E-12 ,   7.759666E-11 ,   2.034960E-08])
heavy_yields_m5p0_z0001 = np.array([9.063641E-06 ,   1.171424E-08 ,   2.788503E-09 ,   5.999624E-09 ,   4.173124E-10 ,   1.176091E-09 ,   6.731594E-10 ,   5.373442E-09 ,   3.016574E-10 ,   4.664019E-10 ,   5.200886E-11 ,   2.883606E-10 ,   7.286195E-11 ,   8.178809E-12 ,   6.911541E-11 ,   4.963688E-09])
# AGB yields from FRUITY at z = 0.0003
heavy_yields_m1p3_z0003 = np.array([2.672576E-05 ,   9.594310E-09 ,   1.560628E-09 ,   3.220023E-09 ,   2.006344E-10 ,   5.783329E-10 ,   3.468567E-10 ,   5.147471E-09 ,   5.211763E-10 ,   1.731746E-09 ,   1.695026E-10 ,   1.028283E-09 ,  2.044587E-10 ,   2.183713E-11 ,   1.871558E-10 ,   1.281200E-07])
heavy_yields_m1p5_z0003 = np.array([2.667789E-05 ,   1.638274E-08 ,   2.835807E-09 ,   5.860083E-09 ,   3.761706E-10 ,   9.938828E-10 ,   5.743317E-10 ,   1.013500E-08 ,   1.033860E-09 ,   3.520406E-09 ,   3.536148E-10 ,   2.082738E-09 ,  4.061634E-10 ,   3.672571E-11 ,   3.449169E-10 ,   2.462733E-07])
heavy_yields_m2p0_z0003 = np.array([2.671957E-05 ,   8.983035E-09 ,   2.259885E-09 ,   5.716426E-09 ,   4.122744E-10 ,   1.010585E-09 ,   5.230674E-10 ,   1.123470E-08 ,   1.186531E-09 ,   4.082100E-09 ,   4.379078E-10 ,   2.333164E-09 ,  4.907328E-10 ,   4.399723E-11 ,   4.296987E-10 ,   2.756212E-07])
heavy_yields_m2p5_z0003 = np.array([2.668789E-05 ,   5.393785E-09 ,   1.394141E-09 ,   3.561403E-09 ,   2.509029E-10 ,   6.939392E-10 ,   3.651996E-10 ,   5.747802E-09 ,   6.095373E-10 ,   2.015857E-09 ,   2.106054E-10 ,   1.223479E-09 ,  2.771576E-10 ,   2.787161E-11 ,   2.565736E-10 ,   1.785759E-07])
heavy_yields_m3p0_z0003 = np.array([2.667508E-05 ,   1.030379E-08 ,   2.728998E-09 ,   7.050640E-09 ,   4.850797E-10 ,   1.392489E-09 ,   7.604108E-10 ,   9.632635E-09 ,   9.462509E-10 ,   2.783880E-09 ,   2.784886E-10 ,   1.786008E-09 ,  4.265937E-10 ,   3.999403E-11 ,   3.853253E-10 ,   1.445154E-07])
heavy_yields_m4p0_z0003 = np.array([2.668628E-05 ,   7.829968E-09 ,   1.816481E-09 ,   4.134825E-09 ,   2.827677E-10 ,   8.069581E-10 ,   4.584193E-10 ,   3.495219E-09 ,   3.067282E-10 ,   8.022671E-10 ,   8.397374E-11 ,   5.337259E-10 ,  1.381087E-10 ,   1.764292E-11 ,   1.404448E-10 ,   4.394903E-08])
heavy_yields_m5p0_z0003 = np.array([2.669641E-05 ,   9.193129E-09 ,   2.066769E-09 ,   4.333943E-09 ,   2.990474E-10 ,   8.263962E-10 ,   4.673368E-10 ,   2.082044E-09 ,   1.414218E-10 ,   2.931184E-10 ,   3.488688E-11 ,   1.954519E-10 ,  5.398237E-11 ,   1.081861E-11 ,   6.518660E-11 ,   9.167519E-09])
# AGB yields from FRUITY at z = 0.001
heavy_yields_m1p3_z001 = np.array([8.948909E-05 ,   1.110631E-08 ,   2.387737E-09 ,   6.466722E-09 ,   4.273378E-10 ,   1.341486E-09 ,   7.733491E-10 ,   1.256720E-08 ,   1.306699E-09 ,   4.646293E-09 ,   4.419624E-10 ,   2.720331E-09 ,  5.572589E-10 ,   6.333267E-11 ,   5.365287E-10 ,   2.141311E-07])
heavy_yields_m1p5_z001 = np.array([8.943435E-05 ,   1.852454E-08 ,   4.064985E-09 ,   1.072026E-08 ,   6.881341E-10 ,   2.011623E-09 ,   1.118744E-09 ,   2.427820E-08 ,   2.643356E-09 ,   9.190655E-09 ,   9.069281E-10 ,   5.350213E-09 ,  1.040186E-09 ,   9.584691E-11 ,   9.037119E-10 ,   3.729063E-07])
heavy_yields_m2p0_z001 = np.array([8.910198E-05 ,   3.742544E-08 ,   8.869695E-09 ,   2.321374E-08 ,   1.531693E-09 ,   3.785491E-09 ,   1.947773E-09 ,   4.680917E-08 ,   5.044193E-09 ,   1.729345E-08 ,   1.852642E-09 ,   9.813511E-09 ,  2.049605E-09 ,   1.716789E-10 ,   1.729104E-09 ,   5.792313E-07])
heavy_yields_m2p5_z001 = np.array([8.913664E-05 ,   2.563683E-08 ,   6.852513E-09 ,   1.906281E-08 ,   1.263687E-09 ,   3.134252E-09 ,   1.359133E-09 ,   3.283245E-08 ,   3.545587E-09 ,   1.194933E-08 ,   1.275845E-09 ,   6.962687E-09 ,  1.536047E-09 ,   1.355959E-10 ,   1.391812E-09 ,   4.388261E-07])
heavy_yields_m3p0_z001 = np.array([8.909280E-05 ,   2.235622E-08 ,   6.054069E-09 ,   1.617839E-08 ,   1.125161E-09 ,   3.297171E-09 ,   1.783887E-09 ,   2.505482E-08 ,   2.540479E-09 ,   7.762937E-09 ,   7.906028E-10 ,   4.910079E-09 ,  1.165020E-09 ,   1.125875E-10 ,   1.050820E-09 ,   3.169538E-07])
heavy_yields_m4p0_z001 = np.array([8.928102E-05 ,   1.368091E-08 ,   3.270397E-09 ,   7.944525E-09 ,   5.486689E-10 ,   1.623672E-09 ,   9.295214E-10 ,   8.413598E-09 ,   8.114168E-10 ,   2.305170E-09 ,   2.447513E-10 ,   1.474377E-09 ,  3.668991E-10 ,   5.124336E-11 ,   3.739967E-10 ,   8.869385E-08])
heavy_yields_m5p0_z001 = np.array([8.931102E-05 ,   1.202101E-08 ,   2.542955E-09 ,   6.036041E-09 ,   3.611955E-10 ,   1.064288E-09 ,   5.881421E-10 ,   2.773067E-09 ,   2.502753E-10 ,   6.587508E-10 ,   8.152390E-11 ,   4.525566E-10 ,  1.293703E-10 ,   3.181817E-11 ,   1.713146E-10 ,   1.800730E-08])
# AGB yields from FRUITY at z = 0.003
heavy_yields_m1p3_z003 = np.array([2.686942E-04 ,   2.943256E-08 ,   6.233489E-09 ,   1.742526E-08 ,   1.149282E-09 ,   3.656068E-09 ,   2.211704E-09 ,   3.557167E-08 ,   3.675941E-09 ,   1.290908E-08 ,   1.178069E-09 ,   7.479672E-09 ,  1.462509E-09 ,   1.710365E-10 ,   1.456710E-09 ,   1.207471E-07])
heavy_yields_m1p5_z003 = np.array([2.683778E-04 ,   5.726331E-08 ,   1.225413E-08 ,   3.591007E-08 ,   2.253464E-09 ,   7.225183E-09 ,   3.923862E-09 ,   9.386292E-08 ,   1.045892E-08 ,   3.666019E-08 ,   3.185488E-09 ,   2.102857E-08 ,  4.222732E-09 ,   3.595853E-10 ,   3.921810E-09 ,   3.327199E-07])
heavy_yields_m2p0_z003 = np.array([2.680550E-04 ,   1.213799E-07 ,   2.660688E-08 ,   7.529306E-08 ,   4.721772E-09 ,   1.418565E-08 ,   7.707440E-09 ,   2.095066E-07 ,   2.324221E-08 ,   8.147899E-08 ,   7.586953E-09 ,   4.650711E-08 ,  8.844217E-09 ,   6.851202E-10 ,   7.555213E-09 ,   6.048155E-07])
heavy_yields_m2p5_z003 = np.array([2.682439E-04 ,   1.118195E-07 ,   2.539439E-08 ,   6.745604E-08 ,   4.390450E-09 ,   1.168810E-08 ,   6.325018E-09 ,   1.704970E-07 ,   1.880025E-08 ,   6.585897E-08 ,   6.766919E-09 ,   3.737921E-08 ,  7.182493E-09 ,   5.808084E-10 ,   5.873880E-09 ,   5.294488E-07])
heavy_yields_m3p0_z003 = np.array([2.683174E-04 ,   4.857390E-08 ,   1.183098E-08 ,   3.130753E-08 ,   2.255900E-09 ,   6.020769E-09 ,   3.274160E-09 ,   6.978970E-08 ,   7.459779E-09 ,   2.599031E-08 ,   2.757099E-09 ,   1.496567E-08 ,  3.127474E-09 ,   3.057519E-10 ,   2.761779E-09 ,   4.173072E-07])
heavy_yields_m4p0_z003 = np.array([2.683535E-04 ,   3.214236E-08 ,   7.634606E-09 ,   1.894865E-08 ,   1.352770E-09 ,   3.795665E-09 ,   2.107493E-09 ,   2.438214E-08 ,   2.484256E-09 ,   7.635099E-09 ,   8.274024E-10 ,   4.497304E-09 ,  1.007902E-09 ,   1.451376E-10 ,   1.012045E-09 ,   1.518425E-07])
heavy_yields_m5p0_z003 = np.array([2.683304E-04 ,   1.953753E-08 ,   4.321667E-09 ,   1.069867E-08 ,   7.419903E-10 ,   2.411470E-09 ,   1.477157E-09 ,   1.030542E-08 ,   1.036289E-09 ,   3.043881E-09 ,   3.542755E-10 ,   1.927153E-09 ,  4.939677E-10 ,   1.042959E-10 ,   5.921954E-10 ,   5.878302E-08])
# AGB yields from FRUITY at z = 0.006
heavy_yields_m1p3_z006 = np.array([5.374971E-04 ,   5.876760E-08 ,   1.252154E-08 ,   3.397799E-08 ,   2.236598E-09 ,   6.899133E-09 ,   4.106743E-09 ,   4.596075E-08 ,   4.447681E-09 ,   1.369608E-08 ,   1.289588E-09 ,   7.597088E-09 ,  1.514203E-09 ,   2.427336E-10 ,   1.657577E-09 ,   3.036912E-08])
heavy_yields_m1p5_z006 = np.array([5.371124E-04 ,   1.480252E-07 ,   3.087835E-08 ,   8.800938E-08 ,   5.593920E-09 ,   1.705407E-08 ,   9.049286E-09 ,   1.797510E-07 ,   1.877019E-08 ,   5.660940E-08 ,   4.780221E-09 ,   2.997057E-08 ,  5.574773E-09 ,   5.104413E-10 ,   5.186250E-09 ,   1.150726E-07])
heavy_yields_m2p0_z006 = np.array([5.363643E-04 ,   3.265268E-07 ,   6.742671E-08 ,   1.876377E-07 ,   1.170298E-08 ,   3.524636E-08 ,   1.863526E-08 ,   3.671265E-07 ,   3.754914E-08 ,   1.106153E-07 ,   9.609244E-09 ,   5.755752E-08 ,  1.028607E-08 ,   8.360980E-10 ,   8.956928E-09 ,   2.293035E-07])
heavy_yields_m2p5_z006 = np.array([5.356982E-04 ,   4.212891E-07 ,   8.830010E-08 ,   2.462052E-07 ,   1.601752E-08 ,   4.696410E-08 ,   2.696820E-08 ,   4.630041E-07 ,   4.437964E-08 ,   1.373251E-07 ,   1.230689E-08 ,   7.158441E-08 ,  1.240969E-08 ,   1.012884E-09 ,   1.066817E-08 ,   2.898261E-07])
heavy_yields_m3p0_z006 = np.array([5.369583E-04 ,   1.482076E-07 ,   3.357458E-08 ,   9.145022E-08 ,   6.006609E-09 ,   1.659278E-08 ,   9.197493E-09 ,   1.976446E-07 ,   2.063818E-08 ,   6.279437E-08 ,   6.487105E-09 ,   3.364596E-08 ,  6.277201E-09 ,   5.838324E-10 ,   5.152575E-09 ,   1.514277E-07])
heavy_yields_m4p0_z006 = np.array([5.369715E-04 ,   1.146862E-07 ,   2.541545E-08 ,   5.726778E-08 ,   4.001893E-09 ,   9.904360E-09 ,   5.367833E-09 ,   5.860122E-08 ,   5.798367E-09 ,   1.760646E-08 ,   1.884464E-09 ,   1.015334E-08 ,  2.111854E-09 ,   2.943582E-10 ,   2.019057E-09 ,   8.539259E-08])
heavy_yields_m5p0_z006 = np.array([5.371054E-04 ,   3.678853E-08 ,   8.024023E-09 ,   2.022229E-08 ,   1.411021E-09 ,   4.510987E-09 ,   2.785740E-09 ,   2.248429E-08 ,   2.302822E-09 ,   7.027607E-09 ,   8.177904E-10 ,   4.296702E-09 ,  1.034480E-09 ,   2.138200E-10 ,   1.213221E-09 ,   4.307849E-08])
# AGB yields from FRUITY at z = 0.010
heavy_yields_m1p3_z010 = np.array([8.960275E-04 ,   4.524181E-08 ,   8.971635E-09 ,   2.275923E-08 ,   1.507430E-09 ,   5.562821E-09 ,   3.746360E-09 ,   1.313040E-08 ,   1.276719E-09 ,   3.423100E-09 ,   4.984375E-10 ,   2.440333E-09 ,  7.692683E-10 ,   2.841757E-10 ,   1.243586E-09 ,   1.318145E-08])
heavy_yields_m1p5_z010 = np.array([8.954069E-04 ,   2.032684E-07 ,   4.114810E-08 ,   1.092109E-07 ,   7.069325E-09 ,   2.040754E-08 ,   1.153742E-08 ,   1.108258E-07 ,   9.683362E-09 ,   2.574519E-08 ,   2.247403E-09 ,   1.293008E-08 ,  2.436507E-09 ,   3.929499E-10 ,   2.675045E-09 ,   3.275532E-08])
heavy_yields_m2p0_z010 = np.array([8.944569E-04 ,   5.296853E-07 ,   1.019616E-07 ,   2.641488E-07 ,   1.637209E-08 ,   4.746837E-08 ,   2.529108E-08 ,   2.972182E-07 ,   2.691108E-08 ,   6.646485E-08 ,   5.638313E-09 ,   3.175802E-08 ,  5.504895E-09 ,   5.904303E-10 ,   5.127010E-09 ,   9.391766E-08])
heavy_yields_m2p5_z010 = np.array([8.936484E-04 ,   7.814950E-07 ,   1.508035E-07 ,   3.921223E-07 ,   2.498457E-08 ,   7.057980E-08 ,   3.952573E-08 ,   3.773476E-07 ,   3.139705E-08 ,   8.114415E-08 ,   6.818475E-09 ,   3.859635E-08 ,  6.550510E-09 ,   6.704002E-10 ,   6.079474E-09 ,   1.314204E-07])
heavy_yields_m3p0_z010 = np.array([8.946699E-04 ,   4.368043E-07 ,   8.795580E-08 ,   2.236082E-07 ,   1.401537E-08 ,   3.948654E-08 ,   2.110826E-08 ,   2.313985E-07 ,   2.071037E-08 ,   5.067806E-08 ,   4.621472E-09 ,   2.459521E-08 ,  4.336417E-09 ,   5.212507E-10 ,   4.035384E-09 ,   8.174216E-08])
heavy_yields_m4p0_z010 = np.array([8.952185E-04 ,   1.036564E-07 ,   2.160683E-08 ,   5.403771E-08 ,   3.667578E-09 ,   1.099986E-08 ,   6.617591E-09 ,   6.152398E-08 ,   5.806326E-09 ,   1.598413E-08 ,   1.706911E-09 ,   9.105461E-09 ,  1.915758E-09 ,   3.683057E-10 ,   2.079216E-09 ,   3.840077E-08])
heavy_yields_m5p0_z010 = np.array([8.953395E-04 ,   6.169792E-08 ,   1.301111E-08 ,   3.279534E-08 ,   2.248522E-09 ,   7.340955E-09 ,   4.644733E-09 ,   3.163514E-08 ,   3.069915E-09 ,   8.451508E-09 ,   1.009671E-09 ,   5.163795E-09 ,  1.260535E-09 ,   3.208435E-10 ,   1.606606E-09 ,   2.286916E-08])
# old elements
# C, Fe, Sr, Y, Zr, Mo, Ba, La, Ce, Nd, Eu, Pb
# new elements
# C, Fe, Sr, Y, Zr, (Nb), Mo, (Ru), Ba, La, Ce, [Pr], Nd, (Sm), Eu, [Dy], Pb

#asplund_heavy_old = np.array([0.0023637158360109672, 0.001262866, 5.05359E-08, 1.12185E-08, 2.69838E-08, 5.66319E-09, 1.6165880945423e-08, 1.36005E-09, 4.14308E-09, 3.46191E-09, 3.91303E-10, 9.06219443353e-09])
asplund_heavy = np.array([0.0023637158360109672, 0.001262866, 5.05359E-08, 1.12185E-08, 2.69838E-08, 2.08462E-09, 5.66319E-09, 4.42132E-09, 1.6165880945423e-08, 1.36005E-09, 4.14308E-09, 5.7512656072E-10, 3.46191E-09, 1.06643E-09, 3.91303E-10, 1.59228702930863e-09, 9.06219443353e-09])

def scale_solar_values(asplund_heavy,input_Z):
    """
    scales solar abundance values by your desired input metallicity.
    """
    solar_Z = 0.0142

    scaled_asplund_heavy = []
    # using input and solar metallicities, scale the asplund values for the elements we desire
    for i in range(len(asplund_heavy)):
        # skip C and Mg, these should be the same - we only need to scale the heavy elems
        if i == 0 or i == 1:
            scaled_asplund_heavy.append(asplund_heavy[i])
            continue
        else:
            scaled_asplund_heavy.append(asplund_heavy[i] * input_Z / solar_Z)
    
    return scaled_asplund_heavy

def select_AGB_yields(AGB_mass, metalZ):
    """
    returns AGB yields from FRUTIY database, sorted by mass and metallicity
    
    input: AGB mass, metallicity
    selects correct AGB yields from fruity model for given parameters
    return: heavy_yields
    """
    # select correct AGB yield file based on accretion model
    if AGB_mass == '1p3': 
        if metalZ == '0001':
            heavy_yields = heavy_yields_m1p3_z0001
        elif metalZ == '0003':
            heavy_yields = heavy_yields_m1p3_z0003
        elif metalZ == '001':
            heavy_yields = heavy_yields_m1p3_z001
        elif metalZ == '003':
            heavy_yields = heavy_yields_m1p3_z003
        elif metalZ == '006':
            heavy_yields = heavy_yields_m1p3_z006
        elif metalZ == '010':
            heavy_yields = heavy_yields_m1p3_z010
    elif AGB_mass == '1p5': 
        if metalZ == '0001':
            heavy_yields = heavy_yields_m1p5_z0001
        elif metalZ == '0003':
            heavy_yields = heavy_yields_m1p5_z0003
        elif metalZ == '001':
            heavy_yields = heavy_yields_m1p5_z001
        elif metalZ == '003':
            heavy_yields = heavy_yields_m1p5_z003
        elif metalZ == '006':
            heavy_yields = heavy_yields_m1p5_z006
        elif metalZ == '010':
            heavy_yields = heavy_yields_m1p5_z010
    elif AGB_mass == '2p0': 
        if metalZ == '0001':
            heavy_yields = heavy_yields_m2p0_z0001
        elif metalZ == '0003':
            heavy_yields = heavy_yields_m2p0_z0003
        elif metalZ == '001':
            heavy_yields = heavy_yields_m2p0_z001
        elif metalZ == '003':
            heavy_yields = heavy_yields_m2p0_z003
        elif metalZ == '006':
            heavy_yields = heavy_yields_m2p0_z006
        elif metalZ == '010':
            heavy_yields = heavy_yields_m2p0_z010
    elif AGB_mass == '2p5': 
        if metalZ == '0001':
            heavy_yields = heavy_yields_m2p5_z0001
        elif metalZ == '0003':
            heavy_yields = heavy_yields_m2p5_z0003
        elif metalZ == '001':
            heavy_yields = heavy_yields_m2p5_z001
        elif metalZ == '003':
            heavy_yields = heavy_yields_m2p5_z003
        elif metalZ == '006':
            heavy_yields = heavy_yields_m2p5_z006
        elif metalZ == '010':
            heavy_yields = heavy_yields_m2p5_z010
    elif AGB_mass == '3p0': 
        if metalZ == '0001':
            heavy_yields = heavy_yields_m3p0_z0001
        elif metalZ == '0003':
            heavy_yields = heavy_yields_m3p0_z0003
        elif metalZ == '001':
            heavy_yields = heavy_yields_m3p0_z001
        elif metalZ == '003':
            heavy_yields = heavy_yields_m3p0_z003
        elif metalZ == '006':
            heavy_yields = heavy_yields_m3p0_z006
        elif metalZ == '010':
            heavy_yields = heavy_yields_m3p0_z010
    elif AGB_mass == '4p0': 
        if metalZ == '0001':
            heavy_yields = heavy_yields_m4p0_z0001
        elif metalZ == '0003':
            heavy_yields = heavy_yields_m4p0_z0003
        elif metalZ == '001':
            heavy_yields = heavy_yields_m4p0_z001
        elif metalZ == '003':
            heavy_yields = heavy_yields_m4p0_z003
        elif metalZ == '006':
            heavy_yields = heavy_yields_m4p0_z006
        elif metalZ == '010':
            heavy_yields = heavy_yields_m4p0_z010
    elif AGB_mass == '4p0': 
        if metalZ == '0001':
            heavy_yields = heavy_yields_m4p0_z0001
        elif metalZ == '0003':
            heavy_yields = heavy_yields_m4p0_z0003
        elif metalZ == '001':
            heavy_yields = heavy_yields_m4p0_z001
        elif metalZ == '003':
            heavy_yields = heavy_yields_m4p0_z003
        elif metalZ == '006':
            heavy_yields = heavy_yields_m4p0_z006
        elif metalZ == '010':
            heavy_yields = heavy_yields_m4p0_z010
    elif AGB_mass == '5p0': 
        if metalZ == '0001':
            heavy_yields = heavy_yields_m5p0_z0001
        elif metalZ == '0003':
            heavy_yields = heavy_yields_m5p0_z0003
        elif metalZ == '001':
            heavy_yields = heavy_yields_m5p0_z001
        elif metalZ == '003':
            heavy_yields = heavy_yields_m5p0_z003
        elif metalZ == '006':
            heavy_yields = heavy_yields_m5p0_z006
        elif metalZ == '010':
            heavy_yields = heavy_yields_m5p0_z010
            
    return heavy_yields

# determine surface abundances from the model output. Use surface data and heavy element AGB yields
def get_surf_abunds(surf_data,heavy_yields,asplund_heavy,scaled_asplund_heavy):
    """
    comptue surface abundances from STARS model output using heavy element AGB yields, 
        asplund abundances, and scaled asplund abundances.
    
    Currently only 'elements of interest' are computed in [X/Fe] format
    C, Mg, Sr, Y, Zr, Mo, Ba, La, Ce, Nd, Eu, Pb
    
    Full surface abunds is _possible_ for the complete pattern, although only for graphical reasons...
        - we have not computed the full elemental pattern here, but it will come!
    """
    C  = (surf_data[44] + surf_data[9] + surf_data[10])
    # sum magnesium 24, 25, 26 from surface data file
    #Mg = (surf_data[19] + surf_data[20] + surf_data[21]) 

    # compute model abundances from AGB yields and asplund mass fractions using arbritrarium (surf_data[30]) 
    Fe = (surf_data[30] * heavy_yields[0]) + ((1 - surf_data[30]) * scaled_asplund_heavy[1])
    Sr = (surf_data[30] * heavy_yields[1]) + ((1 - surf_data[30]) * scaled_asplund_heavy[2])
    Y  = (surf_data[30] * heavy_yields[2]) + ((1 - surf_data[30]) * scaled_asplund_heavy[3])
    Zr = (surf_data[30] * heavy_yields[3]) + ((1 - surf_data[30]) * scaled_asplund_heavy[4])
    Nb = (surf_data[30] * heavy_yields[4]) + ((1 - surf_data[30]) * scaled_asplund_heavy[5])
    Mo = (surf_data[30] * heavy_yields[5]) + ((1 - surf_data[30]) * scaled_asplund_heavy[6])
    Ru = (surf_data[30] * heavy_yields[6]) + ((1 - surf_data[30]) * scaled_asplund_heavy[7])
    Ba = (surf_data[30] * heavy_yields[7]) + ((1 - surf_data[30]) * scaled_asplund_heavy[8])
    La = (surf_data[30] * heavy_yields[8]) + ((1 - surf_data[30]) * scaled_asplund_heavy[9])
    Ce = (surf_data[30] * heavy_yields[9]) + ((1 - surf_data[30]) * scaled_asplund_heavy[10])
    Pr = (surf_data[30] * heavy_yields[10])+ ((1 - surf_data[30]) * scaled_asplund_heavy[11])
    Nd = (surf_data[30] * heavy_yields[11])+ ((1 - surf_data[30]) * scaled_asplund_heavy[12])
    Sm = (surf_data[30] * heavy_yields[12])+ ((1 - surf_data[30]) * scaled_asplund_heavy[13])
    Eu = (surf_data[30] * heavy_yields[13])+ ((1 - surf_data[30]) * scaled_asplund_heavy[14])
    Dy = (surf_data[30] * heavy_yields[14])+ ((1 - surf_data[30]) * scaled_asplund_heavy[15]) 
    Pb = (surf_data[30] * heavy_yields[15])+ ((1 - surf_data[30]) * scaled_asplund_heavy[16])

    # convert to square bracket abundances using asplund solar values
    C_Fe  = np.log10( C / Fe) - np.log10(asplund_heavy[0] / asplund_heavy[1])
    #Mg_Fe = np.log10(Mg / Fe) - np.log10(asplund_heavy[1] / asplund_heavy[2])
    Sr_Fe = np.log10(Sr / Fe) - np.log10(asplund_heavy[2] / asplund_heavy[1])
    Y_Fe  = np.log10( Y / Fe) - np.log10(asplund_heavy[3] / asplund_heavy[1])
    Zr_Fe = np.log10(Zr / Fe) - np.log10(asplund_heavy[4] / asplund_heavy[1])
    Nb_Fe = np.log10(Nb / Fe) - np.log10(asplund_heavy[5] / asplund_heavy[1])
    Mo_Fe = np.log10(Mo / Fe) - np.log10(asplund_heavy[6] / asplund_heavy[1])
    Ru_Fe = np.log10(Ru / Fe) - np.log10(asplund_heavy[7] / asplund_heavy[1])
    Ba_Fe = np.log10(Ba / Fe) - np.log10(asplund_heavy[8] / asplund_heavy[1])
    La_Fe = np.log10(La / Fe) - np.log10(asplund_heavy[9] / asplund_heavy[1])
    Ce_Fe = np.log10(Ce / Fe) - np.log10(asplund_heavy[10]/ asplund_heavy[1])
    Pr_Fe = np.log10(Pr / Fe) - np.log10(asplund_heavy[11]/ asplund_heavy[1])
    Nd_Fe = np.log10(Nd / Fe) - np.log10(asplund_heavy[12]/ asplund_heavy[1])
    Sm_Fe = np.log10(Sm / Fe) - np.log10(asplund_heavy[13]/ asplund_heavy[1])
    Eu_Fe = np.log10(Eu / Fe) - np.log10(asplund_heavy[14]/ asplund_heavy[1])
    Dy_Fe = np.log10(Dy / Fe) - np.log10(asplund_heavy[15]/ asplund_heavy[1])
    Pb_Fe = np.log10(Pb / Fe) - np.log10(asplund_heavy[16]/ asplund_heavy[1])

    surf_abunds = np.array([C_Fe,Sr_Fe,Y_Fe,Zr_Fe,Nb_Fe,Mo_Fe,Ru_Fe,Ba_Fe,La_Fe,Ce_Fe,Pr_Fe,Nd_Fe,Sm_Fe,Eu_Fe,Dy_Fe,Pb_Fe])
   
    return surf_abunds

def split_surf_file(surf_file):
    """
    helper function to extract evolutionary track parameters from file string:
    model name,
    metallicity, 
    AGB mass
    inital mass
    final mass
    """
    model_name = surf_file.split('/')[-1][12:]
    #print(surf_file.split('/'))
    #print(model_name)
    metalZ = model_name.split('_')[-1][1:]
    AGB_mass = model_name.split('_')[-2][3:] #(model_name.split('_')[-2][3:4]+'.'+model_name.split('_')[-2][5:6])
    init_mass = (model_name.split('_')[0][1]+'.'+model_name.split('_')[0][3:5])
    final_mass = (model_name.split('_')[1][1]+'.'+model_name.split('_')[1][3:5])
    #if AGB_mass < init_mass: 
    #    return None
    return metalZ,AGB_mass,init_mass,final_mass

split = [split_surf_file(sf) for sf in list_surf_files if split_surf_file(sf) is not None]
unique = [list(np.unique(x)) for x in np.array(split).T]
#print(unique)
max_len_mod_array = 30000 ## this is how long the plot / surface files are -- how many timesteps are taken in the models? (variable, large number will cover them all but will make the grid much more memory intensive)
num_elements = 16 # see above for number of elements
num_params = 4 # Teff, logg, Fe/H, mass
ndims = tuple(len(u) for u in unique) + (max_len_mod_array, num_elements + num_params)
#print(ndims)

print('Populating the MODEL matrix...')
# create massive array to hold all parameter combinations
MODEL = np.zeros(shape=ndims,dtype=np.float32)
VALID = np.zeros(shape=ndims[:-1],dtype=bool)

for i,(plot_file,surf_file) in enumerate(zip(list_plot_files,list_surf_files)):

    #print(i,plot_file,surf_file)

    # extract evo track params from filename
    metalZ, AGB_mass, init_mass, final_mass = split_surf_file(surf_file)
    #print(metalZ, AGB_mass, init_mass, final_mass)
    
    # apply physically motivated cuts to the grid 
    # this will limit the size of the model grid file and speed up generation of the grid and search
    # no need for empty nan slices, skip them entirely 
    # cut on AGB mass vs initial stellar mass
    if (np.float32(AGB_mass[0]+'.'+AGB_mass[-1])) < np.float32(init_mass):
        continue

    # create indices for the evolutionary track parameters
    a,b,c,d = unique[0].index(metalZ),unique[1].index(AGB_mass),unique[2].index(init_mass),unique[3].index(final_mass)
    # extract evo track / model data
    #acc_data = np.genfromtxt(data_dir+plot_file,dtype=np.float32); acc_data = acc_data.T
    acc_data = read_plot_file_formatted(data_dir+plot_file); acc_data = acc_data.T
    surf_data = np.genfromtxt(data_dir+surf_file,dtype=np.float32); surf_data = surf_data.T
    #print(np.shape(acc_data))
    file_length = np.shape(acc_data)[1]

    # correct spacing of the grid if the files are too large to fit into the 30000 size
    if file_length > max_len_mod_array:
        #print(plot_file,'is too long to fit in the MODEL grid...')
        # compute ratio of file length to size of grid
        ratio = file_length / max_len_mod_array
        round_ratio = np.int32(round(ratio, 0))

        new_spacing = math.ceil(file_length / max_len_mod_array)
        #print(f' Taking every {new_spacing} timestep.')
        acc_data = acc_data[:, ::new_spacing]
        surf_data = surf_data[:, ::new_spacing]
        #print(np.shape(acc_data),'\n')

    # model number from plot file
    model_number = acc_data[0].astype(np.int32); #print(isinstance(model_number,(float,int,list,tuple,set,range,str,dict,bool)))
    
    try:
        n_models = len(model_number)
    except:
        print(plot_file,' has no length \n')
    
    # compute metallicity from metalZ and make into vector with length model number
    model_metallicity = np.log10(np.float32('0.'+metalZ)/0.0142) * np.ones(n_models)# ; print(model_metallicity) 
    # teff from plot file
    model_Teff = acc_data[3]
    # comptued logg from plot file
    model_logg = np.log10((bigG*acc_data[5]*solar_mass_grams)/((solar_radius_cm*10**acc_data[2])**2))
    # extract mass from accretion file
    model_mass = acc_data[5] # units of solar masses
    # select correct AGB yield file based on accretion model
    heavy_yields = select_AGB_yields(AGB_mass, metalZ)
    # scale solar values to the model
    scaled_asplund_heavy = scale_solar_values(asplund_heavy,input_Z=np.float32('0.'+metalZ))
    # compute surface abundances from surface data file and metallicity-scaled solar abundances
    surf_abunds = get_surf_abunds(surf_data,heavy_yields,asplund_heavy,scaled_asplund_heavy)

    # append metallicity, temp, logg, and mass info to abundances
    try:
        # stack into big MODEL array
        MODEL[a,b,c,d,:n_models] = np.vstack([np.array([model_metallicity,model_Teff,model_logg,model_mass]),surf_abunds]).T
        # clarify existance of data
        VALID[a,b,c,d,:n_models] = True
    except:
        print('missing information in ',surf_file)

# option to save some storage space: 
# MODEL = MODEL.astype(np.float32)

# old way of writing .npy files (disguised as .dat...)
print('Writing to files...')
with open('model_grid_full_size.dat','wb') as model_handler:
    np.save(model_handler,MODEL)
with open('valid_grid_full_size.dat','wb') as valid_handler:
    np.save(valid_handler,VALID)

# alternative 1: npz compressed
#np.savez_compressed(
#    'model_grid_full_size.npz',
#    MODEL=MODEL,
#    VALID=VALID
#)
# to be read by 
# data = np.load('model_grid_full_size.npz')
# MODEL = data['MODEL']
# VALID = data['VALID']
#
# alternative 2: HDF5
#import h5py
#
#with h5py.File('model_grid_full_size.h5', 'w') as f:
#    f.create_dataset('MODEL', data=MODEL, compression='gzip')
#    f.create_dataset('VALID', data=VALID, compression='gzip')

print('End of Line')