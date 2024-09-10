# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:50:29 2024

@author: Naina Said
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:45:39 2024

@author: wineuser


"""
import subprocess
import random
import numpy as np
import csv
import scipy.constants
import matplotlib.pyplot as plt
import os

def fly_Simion(inputparticles = ['SIMION_INPUT.FLY2']):
    simion_exe = '"C:\Program Files\SIMION-2020\simion"'
    wb = "Drumsox1.iob"
    output_rec = ".\SIMIONresults_temp.txt"
    nr_files = len(inputparticles)
    filecounter = 1
    print("Flying particles...")
    for particles in inputparticles:
        opts = f" --nogui --noprompt fly --particles={particles} --recording-output={output_rec} --restore-potentials=0 --recording=.\Pos_Angles.rec {wb} --retain=0"
        cmd = simion_exe + opts
        cp = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if cp.returncode != 0:
            print("An error has occured while flying the particles. Simion will be rerun with output to terminal.")
            subprocess.run(cmd)
        print(f"{filecounter} out of {nr_files}")
        filecounter += 1
    print("Done. \nTry to remove Input Particle files.")
    for particles in inputparticles:
        try:
            print(f"Removing {particles}...")
            os.remove(particles)
            print("Done.")
        except:
            print("Tried to remove Simion input particle file for fly'm, but could not find the file.")


def fast_adjust_Simion(invec):
    # with normalization planned, this is ment to be an array of 21 values 
    # between 0 and 1. The values are then scaled to the physical voltages.
    simion_exe = "C:\Program Files\SIMION-2020\simion"
    optics_path = '.\OpticsDrumsox'
    pa_file1 = optics_path + '\obj_18_high-gap-8_D4-bothequal_repel31.pa0' 
    # 4 lenses in pa_file1
    pa_file2 = optics_path + '\TOF_Column_1_LARGER-Z1,Z2,Z3_longer-end_true6.PA0' 
    # 11 lenses, but first lense has to be same value as last lense from 
    # pa_file1, so we treat it like 10 lenses
    pa_file3 = optics_path + '\LENSES_correct2_H,I,J,ToF.pa0'
    # 7 lenses, starting from 2 and going to 8
    print(f"Using this input vector for adjustment of voltages:\n{invec}")
    voltages = norm_voltages(invec)
    print(voltages)
    print("hello")
    cmd1 = f"{simion_exe} --nogui fastadj {pa_file1} 1={voltages[0]},2={voltages[2]},3={voltages[3]},4={voltages[4]},5={voltages[1]}"
    # In front section voltages are 1:Sample, 2:Extractor, 3:Focus, 4:CA, 5:Repeller
    # CA is also first in middle section
    print(f"Fast adjusting voltages in PA files...")
    cp1 = subprocess.run(cmd1, stdout=subprocess.DEVNULL)
    cmd2 = f"{simion_exe} --nogui fastadj {pa_file2} 12={voltages[4]},2={voltages[5]},3={voltages[6]},13={voltages[7]},4={voltages[8]},5={voltages[9]},6={voltages[10]},7={voltages[11]},8={voltages[12]},9={voltages[13]},10={voltages[14]},11={voltages[15]}"
    # In middle section voltages are 12:CA, 2:Z1, 3:Z2, 13:Z3, 4:FA, 5:A, 6:B, 7:C, 8:D, 9:E, 10:F, 11:G
    # G also first element of tof section
    cp2 = subprocess.run(cmd2, stdout=subprocess.DEVNULL)
    cmd3 = f"{simion_exe} --nogui fastadj {pa_file3} 2={voltages[15]},4={voltages[16]},3={voltages[17]},1={voltages[18]}"
    # 2:G, 4:H, 3:I, 1:J=TOF
    # complete table of voltages with position in voltages list: 
    # V[0]:sample, 1:Repeller, 2:Extractor, 3:FOC, 4:CA, 5:Z1, 6:Z2, 7:Z3, 8:FA, 9:A, 10:B, 11:C, 12:D, 13:E, 14:F, 15:G, 16:H, 17:I, 18:J=TOF
    cp3 = subprocess.run(cmd3, stdout=subprocess.DEVNULL)
    if cp1.returncode == 0 and cp2.returncode == 0 and cp3.returncode == 0:
        print("Done.")
    else:
        print("An error occured while adjusting the voltages. Simion runs adjustment again with output to terminal.")
        subprocess.run(cmd1)
        subprocess.run(cmd2)
        subprocess.run(cmd3)



def norm_voltages(invec):
    # define max values for the voltages, so that the input value in [0;1] can 
    # be multiplied by it to yield the correct voltage
    maxvoltages = [1000, 10000, 10000, 4000, 10000, 10000, 10000, 5000, 5000, 5000, 5000, 5000, 2000, 2000, 2000, 2000, 1000, 1000, 200]
    if len(maxvoltages) != len(invec):
        raise ValueError(f"Incorrect number of input parameters. There are {len(invec)} elements in the input vector, but it should be {len(maxvoltages)} elements.")
    for c, i in enumerate(invec):
        if abs(i) > 1:
            raise ValueError(f"Incorrect input. Voltages are normalized to [-1,1] but inputvector contains {i} on position {c}")
    involtages = [i1 * i2 for i1, i2 in zip(maxvoltages, invec)]
    return involtages


def reverse_norm_voltages(involtages):
    maxvoltages = [1000, 10000, 10000, 4000, 10000, 10000, 10000, 5000, 5000, 5000, 5000, 5000, 2000, 2000, 2000, 2000, 1000, 1000, 200]
    invec = [i1 / i2 for i1, i2 in zip(involtages,maxvoltages) ]
    return invec


### Funnctions for generating particle starting coordinates
def make_pair_even(x,y,spotsize):
    if (int(x) + int(y)) % 2 == 0:
        return y
    elif y + 1 >= spotsize:
        y = y - int(y)
    else:
        y += 1
    return y

def get_digit(number, digit):
    number = int(number * 10 **(digit-1) % 1 * 10)
    return number

def plusminus():
    return 1 if random.random() < 0.5 else -1

def even_pair(scale=0, maxint=9):
    x = random.randint(0,maxint)
    if x % 2 == 0: 
        y = random.randint(0,int(maxint/2)) * 2
    else:
        y = random.randint(0,int(maxint/2)) * 2 + 1        
    return x * 10 ** scale, y * 10 ** scale

def position_generator_chessy(spotsize=0.001):
    if spotsize > 5:
        spotsize = 5
    on_spot = False
    while not on_spot:
        x,y = even_pair(maxint=int(spotsize))
        for i in range(-1,-4,-1):
            if 10 ** i  > spotsize:
                continue
            a,b = even_pair(i)
            x += a
            y += b
        x += random.random() * 1e-3
        y += random.random() * 1e-3
        if x <= spotsize and y <= spotsize:
            on_spot = True
    return x,y



def create_simple_chessy_PEEM_particle(spotsize=5, e_kin= 180, offset=[0,0], max_angle=25):
    # Spotsize float in mm, Kinetic energy of the particle float in eV, offset of center in mm list of floats [x,y], exit angel float in degree
    # real space checkboard pattern with default emission angles up to +- 25 degree 
    # checkboard is 5x5 1mm square edges largest
    # then 10X10 100um edges
    # then 10x10 10um edges
    # and smallest 10x10 1um edges
    # spotsize will reduce the number of squares accordingly
    x,y = position_generator_chessy(spotsize)
    x = x + offset[0] - spotsize / 2
    y = y + offset[1] - spotsize / 2
    k_max = np.sqrt(2 * scipy.constants.m_e * e_kin * scipy.constants.eV) / scipy.constants.hbar * np.sin(max_angle / 360 * 2 * np.pi) / 1e10
    kx = random.uniform(-k_max, k_max)
    ky = random.uniform(-np.sqrt(k_max**2 - kx**2), np.sqrt(k_max**2 - kx**2))
    return x, y, e_kin, kx, ky


def split_particles_file(no_of_particles):
    # Simion can only handle particle files up 30000(?), so if more particles are used at the same time, input has to be split into multiple input files.
    no_of_files = no_of_particles / 25000
    if no_of_files == int(no_of_files):
        no_of_files = int(no_of_files)
    else:
        no_of_files = int(no_of_files) + 1
    inputfile_names = []
    for i in range(no_of_files):
        inputfile = f"SIMION_INPUT{i}.FLY2"
        inputfile_names.append(inputfile)
    return inputfile_names, no_of_files


def write_fly(no_of_particles = 25000, spotsize=2, max_angle=10):
    if no_of_particles > 25000:
        inputfile_names, no_of_files = split_particles_file(no_of_particles)
    else:
        inputfile_names = ["SIMION_INPUT.FLY2"]
        no_of_files = 1
    for file in range(no_of_files):
        inputfile = inputfile_names[file]
        if file + 1 == no_of_files and (no_of_particles - 25000 * (no_of_files-1)) != 25000:
            particles = no_of_particles % 25000
        else:
            particles = 25000
        with open(inputfile, 'w') as f:
            f.write("particles {\n")
            for n in range(int(particles)):
                x, y, e_kin, kx, ky = create_simple_chessy_PEEM_particle(spotsize=spotsize, max_angle=max_angle, e_kin= 180)
                az = np.arcsin( kx * 1e10 * scipy.constants.hbar /np.sqrt(2 * scipy.constants.m_e * e_kin * scipy.constants.eV)) / np.pi * 180
                el = np.arcsin( ky * 1e10 * scipy.constants.hbar /np.sqrt(2 * scipy.constants.m_e * e_kin * scipy.constants.eV)) / np.pi * 180
                particle_definition = f"standard_beam {{\n\tke = {e_kin}, \t\t-- eV\n\taz = {az},\n\tel = {el},\n\tposition = vector( 1, {y}, {x} ),\t\t-- mm\n}},\n"
                f.write(particle_definition)
            f.write("}")
    print(f"Created {no_of_files} new FLY2 file with {no_of_particles} particles in total. Maximum number of particles per file is 25000.")
    return inputfile_names


def sort_simion_result(simion_results='SIMIONresults_temp.txt'):
    # simion_results is the .txt file which is the output of calling Simions fly function.
    # The particles that hit the detector are stored in matrix res with TOF, PositionX and PositionY for each electron.
    # the ratio of electrons that hit the detector to the total number of electrons created is calculated.
    # The .txt file is then deleted.
    file = simion_results
    res = [[],[],[],[],[],[],[],[]]
    hits_on_electrodes = 0
    hits_on_detector = 0
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) != 9:
               continue
            elif row[1] == '1':
                # start parameters
                # note: x and y correspond to simions y and z coordinates
                x_start = float(row[4])
                y_start = float(row[5])
                #TODO: change .rec file and include angles for KPEEM image tracing particles
                # This will lead to two more elements in row and res
                # Will have to check for position of all elements in row again
                el_start = float(row[6])
                az_start = float(row[7])
                ekin_start = float(row[8])
            elif row[1] == '16':
                #save TOF, position x, position y
                tof = float(row[2])
                x = float(row[4])
                y = float(row[5])
                hits_on_detector += 1
                # restricts to active area of dld8080 detector, 
                # resolution (pixelsize) is 97.96 um (3 pixels with 2x2 pixel hardware binning) -> in SIMION would be 0.09796
                # pixelsize of one pixel is 32.65 um -> 0.03265
                # this results in 80 mm / 0.03265 mm ~= 2450 bins
                
                if np.sqrt(x ** 2 + y ** 2) > 40: 
                    hits_on_detector -=1
                    hits_on_electrodes +=1
                res[0].append(tof)
                res[1].append(x)
                res[2].append(y)
                res[3].append(x_start)
                res[4].append(y_start)
                res[5].append(el_start)
                res[6].append(az_start)
                res[7].append(ekin_start)
            elif row[1] == '4':
                hits_on_electrodes += 1
    print(f"{hits_on_detector} particles hit the detector, {hits_on_electrodes} particles hit some electrodes and therefore did not hit the detector.\nSorted all hits on detector in matrix with time of flight, y and z.")
    ratiohits = hits_on_detector / (hits_on_detector + hits_on_electrodes)
    try:
        os.remove(simion_results)
    except:
        print("Tried to remove Simion output file from fly'm, but could not find the file.")
    return res, ratiohits

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1

    return path


def save_simion_res(res, outputfile = '.\SavedRuns\Simion_histogram_data.txt', invec=[], append_to_file=False):
    # Saves results from results matrix to text file.
    # if no file to write to exists, a new file is created. Otherwise, the existing file is appended.
    header = "---Results from Simion simulation, saved only hits on Detector.-----------\nTOF \tXend \tYend \tXstart \tYstart \telvStart \tazmStart\n"
    try:
        outf = open(outputfile, 'x')
        outf.write(header)
        outf.close()
        print(f"Created new file for results. Name is {outputfile}...")
    except:
        if append_to_file:
            print(f"Add results to existing file {outputfile}...")
        else:
            outputfile = uniquify(outputfile)
    with open(outputfile, 'a') as outf:
        if not invec ==[]:
            outf.write('---input vector lense voltages: ' + str(invec) + '\n')
            outf.write(header)
        for i in range(len(res[0])):
            outf.write(str(res[0][i]) + '\t' + str(res[1][i]) + '\t' + str(res[2][i]) + '\t' + str(res[3][i]) + '\t' + str(res[4][i]) + '\t' + str(res[5][i]) + '\t' + str(res[6][i]) + '\n')
    print(f"Saved 3D simulated data to {outputfile}.")


def make_hist_kx_ky(res, bins=400):
    #if no input is given for energy slice, all energy is integrated
    #TODO: make an energy slice possible
    x = res[1]
    y = res[2]
    fig, ax = plt.subplots()
    h = ax.hist2d(x,y,bins=bins) 
    fig.colorbar(h[3], ax=ax)
    plt.show()


def make_e_kx(res):
    # res is matrix with TOF, position x and position y
    # first, along kx direction
    tof = res[0]
    kx = res[1]
    fig, ax = plt.subplots()
    ax.hist2d(tof, kx, bins= (100,600))
    plt.show()


def bin_simulation(res):
    # this should take the particle coordinates from the simulation and return them as binned results
    # Bin size in xy comes from pixel size of detector (32.65 um but 2x2 hardware binning -> 65.3 um per bin)
    # TODO: determine correct Bin size in energy 
    res = np.array(res)
    res_transp = res.transpose()
    tof = res[0]
    if len(tof)==0:
        print('no Electrons on Detector')
        return
    tofmin = min(tof)
    tofmax = max(tof)
    xybinsize = 0.0653
    xybins = 400 # int(80 // xybinsize)
    tofbinsize = 1e-4
    tofbins = 1#int((tofmax-tofmin) / tofbinsize)    
    h, edges = np.histogramdd(res_transp, bins=(tofbins, xybins, xybins), range=((tofmin,tofmax), (-40,40), (-40,40)))
    return h, edges


def scaling_factors_real(results):
    # evaluation of simulation run by comparing start parameters of each electron to detector position
    # works for real space images 
    # returns list of scaling factor between x,y start position and x,y end position for each electron
    results = list(map(list, zip(*results)))
    sf = []
    for electron in results:
        x_start = electron[3]
        x_end = electron[1]
        y_start = electron[4]
        y_end = electron[2]
        sfx = abs(x_end/x_start)
        sfy = abs(y_end/y_start)
        sftotal = (sfx + sfy) / 2

        sf.append(sftotal)
    return sf

def angle_to_k(angle, energy):
    # angle in degree, energy in eV
    k = np.sqrt(2 * scipy.constants.m_e * energy * scipy.constants.eV) / scipy.constants.hbar * np.sin(angle / 360 * 2 * np.pi) / 1e10
    return k


def scaling_factors_rez(results):
    # evaluation of simulation run by comparing start parameters of each electron to detector position
    # works for momentum (reciproce) space images 
    # returns lists of scaling factor between azm,elv start angles and x,y end position for each electron
    # separate list for azm and elv
    results = list(map(list,zip(*results)))
    sfkx = []
    sfky = []
    for electron in results:
        # elevation influences Simion y coordinate, python x coordinate
        azm = electron[6]
        elv = electron[7]
        x_end = electron[1]
        y_end = electron[2]
        sfelv = elv / x_end
        sfazm = azm / y_end
        sfkx.append(sfelv)
        sfky.append(sfazm)
        
    sf_total = []
    for e in sfky:
        if e <0:
            e = -e
        sf_total.append(e)
    for e in sfkx:
        if e <0:
            e = -e
        sf_total.append(e)
    return sf_total


def eval_sf(sf, ratio, showplot=False):
    if ratio < 0.4:
        grade = -1
        print(f'Insufficient number of particles hit the detector, ratio is {ratio}. Grade of simulation is {grade}')
    else:
        upperlimit = np.nanpercentile(sf,99)
        lowerlimit = np.nanpercentile(sf,1)
        for c,v in enumerate(sf): 
            if v < lowerlimit or v > upperlimit:
                sf[c] = float("NaN")
        sigma = np.nanstd(sf)
        mu = np.nanmean(sf)
        minsf = np.nanmin(sf)
        maxsf = np.nanmax(sf)
        print(f'Scaling factors range from {minsf} to {maxsf}.')
        print(f'Mean of electron scaling factor is {mu}, standard deviation is {sigma}.')
        relsigma = sigma/mu
        if relsigma < 1:
            grade =  1 - relsigma
        else:
            grade = 1/relsigma - 1
        print(f'Grade of simulation is {grade}')
        if showplot == True:
            fig, ax = plt.subplots()
            ax.hist(sf, bins=400)
            plt.show()
    return grade



if __name__ == '__main__':
    #standard_voltages_rez = [90, 8000, 400, 1200, 2600, 1600, 600, 700, 3000, 600, 500, 1500, 300, 150, 200, 50, 50, 30, 80, 20, 30]
    # V[0]:sample, 1:Repeller, 2:Extractor, 3:FOC, 4:CA, 5:Z1, 6:Z2, 7:Z3, 8:FA, 9:A, 10:B, 11:C, 12:D, 13:E, 14:F, 15:G, 16:H, 17:I, 18:J=TOF
    #standard_PEEM_Drumsox = [180, 6000, 6000, 2000, 1200, 2500, 5000, 1100, 800, 338, 2225, 230, 170, 195, 92, 30, 47, 37, 20]
    #settingPEEM_1 = [180, 8000, 8000, 2000, 1100, 2000, 5200, 1100, 600, 360, 2120, 103, 5, 350, 96, 30, 47, 37, 20] # 02. Nov 2023, 06:39h
    settingPEEM_11 = [180, 8000, 8000, 1200, 1100, 2000, 5200, 1100, 600, 360, 2025, 200, 40, 350, 96, 30, 47, 37, 20] # 
    #settingPEEM_2 = [180, 0, 7000, 815, 1800, 3000, 3500, 1000, 600, 850, 1450, 250, 120, 83, 35, 19, 300, 30, 20]# experimental settings, not tested in simulation yet
    #kPEEM = [180, 0, 7000, 1030, 1800, 3000, 3500, 1000, 600, 450, 2300, 400, 230, 83, 35, 11, 30, 30, 20]
    inputvoltages = reverse_norm_voltages(settingPEEM_11)
    #inputvoltages = [0.00429636, 0.01843464, 0.0, 0.39480484, 0.0, 0.45632547, 0.0, 0.0, 0.7122481, 0.0, 0.23248222, 1.0, 0.4844688, 0.0, 1.0, 0.0, 0.0, 0.31254297, 0.0]
    #inputvoltages=[ 0.07660286,  0.96801717,  0.68636438,  0.45254327,  0.12261083,  0.23077575, 0.44449899,  0.13855801,  0.17248154, 0.04777522,  0.63343185,  0.11595436,0.05039628,  0.18052628, 0.00132337,  0.02716226,  0.07692517, 0.09874546,0.07524455]
    #inputvoltages=[1.,0.8093339,0.7197765 , 0.,0.6096807,0.31134427, 0.15109567, 0.2754885 , 0.52147394, 0.66190356, 0.,0.2734777, 0.,0.64896345, 0.,0.17761876, 0.,0.5851489, 0.01133149]
    #inputvoltages = [0.3278697901942431, 0.5482949176166378, 0.7626960715732767, 0.9424703400898843, 0.11686906621458248, 0.4864525437236311, 0.7464446251427307, 0.5463607725017386, 0.3513853292752195, 0.8317104688246062, 0.25101093560865395, 0.9375882813327037, 0.8626711391348817, 0.05798617839000264, 0.4115772306845401, 0.7132618215920403, 0.600655804961327, 0.5765150442951492, 0.6552760505259752]
    # inputvoltages = [0.5744462809887811, 0.8642235215947945, 0.9339971200026542, 0.4509319205779132, 0.07517664295092952, 0.5667831890038717, 0.32463116176348294, 0.2311190045734115, 0.13086784701397203, 0.8747347973798755, 0.6109128146974787, 0.5952595717407858, 0.47637555689900546, 0.9921812501892256, 0.18963431915447748, 0.008372008584550894, 0.7610718153717184, 0.42743533948346024, 0.20102064204867964]
    #inputvoltages = [0.5382572213065943, 0.029061763972254906, 0.5321718108884811, 0.8793988437085241, 0.9349011817205375, 0.3337337063813006, 0.5762063720356484, 0.3396350706576584, 0.28855556257217263, 0.8801200461271489, 0.8548525160060919, 0.2507554387420615, 0.9356435747460745, 0.7996336077954976, 0.7123536654205681, 0.6170195796887389, 0.16889665918128904, 0.2361825297175838, 0.4232262954817273]
    #inputvoltages =[0.189, 0.76, 0.76, 0.285, 0.1045, 0.21, 0.546, 0.209, 0.126, 0.0756, 0.38475, 0.042, 0.019, 0.16625, 0.0456, 0.01425, 0.04465, 0.03885, 0.105]

    #inputvoltages=[0.171, 0.76, 0.80428742, 0.285, 0.1045, 0.19, 0.494, 0.209, 0.126, 0.0684, 0.38475, 0.038, 0.019, 0.16625, 0.0456, 0.01425, 0.04465, 0.03515, 0.095]


    fast_adjust_Simion(inputvoltages)
    inputfilenames = write_fly(1000, spotsize=1, max_angle=10)
    fly_Simion(inputfilenames)
    results, ratiohits = sort_simion_result('SIMIONresults_temp.txt')
    sf = scaling_factors_real(results)
    eval_sf(sf,ratiohits,showplot=True)
    #make_hist_kx_ky(results)
    #save_simion_res(results, invec=inputvoltages)