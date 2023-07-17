# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 08:09:43 2022

@author: matfleisch
"""

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import inspect
import PoreFunctions

def get_filepaths_from_directory(directory, extension, prefix=None):
    """
    Returns a list of filepaths in the given folder with the given extension.
    Filter out all files which do not have the prefix in their filename

    Keyword arguments:
        directory (str): Path to the directory which should be used
        extension (str): Extension of files
        prefix (str): Prefix to filter out files (default: None)
    """
    file_paths = glob.glob('{}\*.{}'.format(directory, extension))
    if prefix:
        file_paths = [file_path for file_path in file_paths if os.path.split(file_path)[1].startswith(prefix)]
    return natsorted(file_paths)

def get_filename_from_path(file_path, prefix=None):
    """
    Returns the filename from the given file_path and removes the prefix if
    given
    """
    filename = os.path.basename(file_path)
    without_extension = os.path.splitext(filename)[0]
    if prefix and without_extension.startswith(prefix):
        return without_extension[len(prefix):]
    else:
        return without_extension

def full_simulation_data(file_path, ext_v, ext_h, delimiter=' '):
    df = pd.read_csv(file_path, delimiter=delimiter, skiprows=4)
    force = np.abs(df['Force'])

    lengths_v = []
    for i in range(len(force)):
        average_distance = 0
        for exts in ext_v:
            x0 = df['P{}-x'.format(exts[0])].iloc[i]
            y0 = df['P{}-y'.format(exts[0])].iloc[i]
            x1 = df['P{}-x'.format(exts[1])].iloc[i]
            y1 = df['P{}-y'.format(exts[1])].iloc[i]
            distance = np.linalg.norm(np.array([x0, y0])-np.array([x1, y1]))
            average_distance += distance
        average_distance /= len(ext_v)
        lengths_v.append(average_distance)
    lengths_v = np.array(lengths_v)
    strain_v = (lengths_v-lengths_v[0])/lengths_v[0]

    lengths_h = []
    for i in range(len(force)):
        average_distance = 0
        for exts in ext_h:
            x0 = df['P{}-x'.format(exts[0])].iloc[i]
            y0 = df['P{}-y'.format(exts[0])].iloc[i]
            x1 = df['P{}-x'.format(exts[1])].iloc[i]
            y1 = df['P{}-y'.format(exts[1])].iloc[i]
            distance = np.linalg.norm(np.array([x0, y0])-np.array([x1, y1]))
            average_distance += distance
        average_distance /= len(ext_h)
        lengths_h.append(average_distance)
    lengths_h = np.array(lengths_h)
    strain_h = (lengths_h-lengths_h[0])/lengths_h[0]

    poissons_ratio = -strain_h/strain_v
    
    with open(file_path) as f:
        length, width, thickness = [next(f) for x in range(3)]
    length = float(length.split(' ')[1].strip('\n'))
    width = float(width.split(' ')[1].strip('\n'))
    thickness = float(thickness.split(' ')[1].strip('\n'))
    stress = force/(width*thickness)

    with open(file_path) as f:
        pfs = eval([next(f) for x in range(4)][-1].split('Parameters: ')[-1])
        param_dict = {}
        for i, pf in enumerate(pfs):
            fname = pf[0]
            parameters = pf[1][3:]
            px = pf[1][0]
            py = pf[1][1]
            a = pf[1][2]
            f = getattr(PoreFunctions, fname)
            fparams = inspect.getfullargspec(f).args[:-4]
            pore_dict = {}
            for p, n in zip(parameters, fparams):
                pore_dict[n] = p
            pore_dict['PosX'] = px
            pore_dict['PosY'] = py
            pore_dict['RotAngle'] = a
            param_dict[i+1] = pore_dict

    results = pd.DataFrame({'Displacement': np.abs(df['Displacement']),
                            'Force': np.abs(df['Force']),
                            'Stress': stress,
                            "Poisson's Ratio": poissons_ratio,
                            'Strain (longitudinal)': np.abs(strain_v),
                            'Strain (transversal)': strain_h,
                            'Lengths (longitudinal)': lengths_v,
                            'Lengths (transversal)': lengths_h})

    return (results, param_dict)

def full_simulation_properties(data, modulus_range=[0., 0.25], poisson_at_strain=0.025):
    temp = data[(data['Strain (longitudinal)']>=modulus_range[0])&(data['Strain (longitudinal)']<=modulus_range[1])]
    fit = np.polyfit(temp['Strain (longitudinal)'], temp['Stress'], 1)
    idx = data['Strain (longitudinal)'].sub(poisson_at_strain).abs().idxmin()
    nu_21 = data.iloc[idx]["Poisson's Ratio"]
    return {'E2': fit[0], 'NU21': nu_21}

def orthotropic_elastic_properties(matrix_txt, delimiter=','):
    """
    Calculates the elastic properties of an orthotropic tensor
    """
    data = np.genfromtxt(matrix_txt, delimiter=delimiter, skip_footer=1)
    data = np.linalg.inv(data)
    e_1 = 1/data[0,0]
    e_2 = 1/data[1,1]
    e_3 = 1/data[2,2]
    g_12 = 1/data[3,3]
    g_13 = 1/data[4,4]
    g_23 = 1/data[5,5]
    nu_12 = -e_2*data[0,1]
    nu_21 = -e_1*data[1,0]
    nu_31 = -e_3*data[2,0]
    nu_13 = -e_1*data[0,2]
    nu_32 = -e_3*data[2,1]
    nu_23 = -e_2*data[1,2]
    with open(matrix_txt, 'r') as f:
        pfs = eval([next(f) for x in range(7)][-1].split('Parameters: ')[-1])
        param_dict = {}
        for i, pf in enumerate(pfs):
            fname = pf[0]
            parameters = pf[1][3:]
            px = pf[1][0]
            py = pf[1][1]
            a = pf[1][2]
            f = getattr(PoreFunctions, fname)
            fparams = inspect.getfullargspec(f).args[:-4]
            pore_dict = {}
            for p, n in zip(parameters, fparams):
                pore_dict[n] = p
            pore_dict['PosX'] = px
            pore_dict['PosY'] = py
            pore_dict['RotAngle'] = a
            param_dict[i+1] = pore_dict

    return ({'E1': e_1, 'E2': e_2, 'E3': e_3, 'G12': g_12, 'G13': g_13, 'G23': g_23,
            'NU12': nu_12, 'NU21': nu_21, 'NU31': nu_31, 'NU13': nu_13, 'NU32': nu_32, 'NU23': nu_23}, param_dict)

# Points used as extensometers

# Hippopede
ext_v = [[1, 4], [2, 3]]    # vertical (y)
ext_h = [[1, 2], [3, 4]]    # horizontal (x)

# Sphylinder Multi
ext_v = [[1, 3]]
ext_h = [[2, 4]]

folders = {
            # 'Example': [r'Results\01_Example', orthotropic_elastic_properties, 'txt'],
            # 'Squircle': [r'Results\02_Squircle', orthotropic_elastic_properties, 'txt'],
            # 'Antichiral': [r'Results\03_Antichiral-Standard', orthotropic_elastic_properties, 'txt'],
            # 'AntichiralSphylinder': [r'Results\04_Antichiral-Sphylinder', orthotropic_elastic_properties, 'txt'],
            # 'AntichiralSphylinder': [r'C:\temp\SPHYLINDER', orthotropic_elastic_properties, 'txt'],
            # 'Superellipse': [r'C:\temp\SUPERELLIPSE', orthotropic_elastic_properties, 'txt'],
            # 'Superellipse2': [r'C:\temp\SUPERELLIPSE2', orthotropic_elastic_properties, 'txt'],
            'SphylinderExp': [r'C:\temp\SPHYLINDER', orthotropic_elastic_properties, 'txt'],
            # 'Nsided': [r'C:\temp\NSIDED', orthotropic_elastic_properties, 'txt'],
            # 'Hippopede': [r'Results\05_Hippopede', full_simulation_properties, 'dat'],
            # 'Sphylinder-Multi': [r'Results\06_SphylinderMulti', full_simulation_properties, 'dat'],
            # 'Superellipse': [r'C:\temp\TEST', orthotropic_elastic_properties, 'txt'],           
            # 'SphylinderMultiS': [r'Results\07_SphylinderMultiS', full_simulation_properties, 'dat'],
            # 'SphylinderMultiS2': [r'Results\08_SphylinderMultiS2', full_simulation_properties, 'dat'],
            # 'Sphylinder-Multi-S': [r'C:\Users\matfleisch\Documents\Messungen\Multimaterial\Simulationen\S6', full_simulation_properties, 'dat'],
            # 'Sphylinder-Multi-SUC': [r'C:\Users\matfleisch\Documents\Messungen\Multimaterial\Simulationen\SUC', orthotropic_elastic_properties, 'txt'],
            # 'Sphylinder-Multi-UCPS': [r'C:\Users\matfleisch\Documents\Messungen\Multimaterial\Simulationen\UCPS\S_0-0', orthotropic_elastic_properties, 'txt'],
            # 'Sphylinder-Multi-T': [r'C:\Users\matfleisch\Documents\Messungen\Multimaterial\Simulationen\T', full_simulation_properties, 'dat'],
            # 'Sphylinder-Multi-N': [r'C:\Users\matfleisch\Documents\Messungen\Multimaterial\Simulationen\N', full_simulation_properties, 'dat'],
}

modulus_range = [0.05/100., 1./100.]
poisson_at_strain = 2.0/100.

save_individual = False

result_dict = {}
for name, (folder, func, ext) in folders.items():
    files = get_filepaths_from_directory(folder, ext)
    results = []
    for i, file in enumerate(files):
        print('{}/{}'.format(i+1, len(files)))
        fname = get_filename_from_path(file)
        if func == full_simulation_properties:
            data, param_dict = full_simulation_data(file, ext_v, ext_h)
            result = full_simulation_properties(data, modulus_range=modulus_range,
                                                poisson_at_strain=poisson_at_strain)
            if save_individual:
                result_file = os.path.join(folder, '{}_{}-strain.xlsx'.format(name, i))
                data.to_excel(result_file, index=False)
        elif func == orthotropic_elastic_properties:
            result, param_dict = orthotropic_elastic_properties(file)
        result['Name'] = fname
        results.append(result)
        for i in range(len(param_dict)):
            p = i+1
            params = param_dict[p]
            for pn, val in params.items():
                result['P{}-{}'.format(p, pn)] = val
    df = pd.DataFrame(results)
    # df = df[df['E2']>0.001]
    result_dict[name] = df
    if len(df) > 0:
        result_file = os.path.join(folder, 'results-{}.xlsx'.format(name))
        df.to_excel(result_file, index=False)

# for name, df in result_dict.items():
#     print('{}: {}, {}'.format(name, df['NU12'].min(), df['NU21'].min()))




