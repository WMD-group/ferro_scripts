#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,splrep,splev
from scipy.optimize import minimize
from copy import deepcopy
import datetime
import yaml
import sys
import os

now=datetime.datetime.now()

def hysteresis_loop(q_energy, energy, celldims, q_chi, chi, E_max, num_E_samples, Ps, debug):
    """
    Plots hysteresis loop given configuration-dependent susceptibility and bulk
    energy data.
    Parameters
    ==========
    q_energy: array(float)
        Collective coordinate values corresponding to data in energy variable.
        Values must be between 0 and 1.
    energy: array(float)
        In Hartrees. Bulk energy per unit cell for atomic configurations
        corresponding to q_energy entries.
    celldims: array(float)
        In bohr. Dimensions of unit cell.
    q_chi: array(float)
        Collective coordinate values corresponding to data in chi variable.
        Values must be between 0 and 1.
    chi: array(float)
        In esu. Electronic contribution to dielectric susceptibility for
        configurations along path between ferroelectric configurations.
    E_max: float
        In kV/cm. Max value of electric field at which equilibrium polarisation
        value should be calculated.
    num_E_samples: int
        Number of electric field samples to plot.
    Ps: float
        In uC/cm^2. Spontaneous polarisation of ferroelectric.
    debug: boolean
        If True, contributions to minimum free energy will be printed to stdout.
    """
    E_max=np.abs(E_max)
    E_asc=np.linspace(-E_max,E_max,num_E_samples)
    E_des=np.linspace(E_max,-E_max,num_E_samples)

    P_asc=get_pol_vs_e(q_energy,energy,celldims,q_chi,chi,E_asc,Ps,debug)
    P_des=get_pol_vs_e(q_energy,energy,celldims,q_chi,chi,E_des,Ps,debug)
    analyze_energy_chi(q_energy, energy, celldims, q_chi, chi)

    fg=plt.figure()
    ax=fg.add_subplot(111)
    ax.plot(E_asc,P_asc,'rx-',label='ascending')
    ax.plot(E_des,P_des,'bx-',label='descending')
    ax.set_xlabel(r'$E(kV/cm)$')
    ax.set_ylabel(r'$P(\mu C/cm^2)$')
    ax.legend()
    global now
    timestamp=now.strftime('%Y%m%d_%H%M%S')
    fg.savefig(os.path.join('output','hysteresis_loop_'+timestamp))

def plot(x_data,y_data,data_label,interp):
    fg=plt.figure()
    ax=fg.add_subplot(111)
    ax.plot(x_data,y_data,'bx',label=r'{} data'.format(data_label))
    x_min=np.min(x_data)
    x_max=np.max(x_data)
    x_range=np.linspace(x_min,x_max,len(x_data)*10)
    ax.plot(x_range,interp(x_range),'r.',label=r'{} interpolated'.format(data_label))
    return ax

def analyze_energy_chi(q_energy, energy, celldims, q_chi, chi):
    """
    Plots inputted energy and linear susceptibility vs collective coordinate
    data with interpolating functions used to generate hysteresis loop. Also
    prints energy barrier information.
    Parameters
    ==========
    q_energy: array(float)
        Collective coordinate values corresponding to data in energy variable.
        Values must be between 0 and 1.
    energy: array(float)
        In Hartrees. Bulk energy per unit cell for atomic configurations
        corresponding to q_energy entries.
    celldims: array(float)
        In bohr. Dimensions of unit cell.
    q_chi: array(float)
        Collective coordinate values corresponding to data in chi variable.
        Values must be between 0 and 1.
    chi: array(float)
        In esu. Electronic contribution to dielectric susceptibility for
        configurations along path between ferroelectric configurations.
    """
    energy=deepcopy(energy)
    q_energy=deepcopy(q_energy)

    e_min=np.min(energy)
    e_max=np.max(energy)
    energy-=e_min
    e_poly=np.polyfit(q_energy,energy/np.prod(celldims),6)
    barrier=e_max-e_min
    print('Barrier height:\n{0:.3e} hartrees/unit cell\n{1:.3e} eV/unit cell\n{2:.3e} hartrees/a0^3'.format(barrier,barrier*27.21,barrier/np.prod(celldims)))

    def eval_e(q):
        return np.polyval(e_poly,q)
    ax=plot(q_energy,energy/np.prod(celldims),'$E$',eval_e)
    ax.set_xlabel(r'$Q$')
    ax.set_ylabel(r'$E(Q)(E_\mathrm{H}/a_0^3$)')
    fg=ax.figure
    timestamp=now.strftime('%Y%m%d_%H%M%S')
    fg.savefig(os.path.join('output','energy_'+timestamp))

    # Sort chi(q) by ascending q.
    sort_idx=np.argsort(q_chi)
    chi=deepcopy(chi)
    q_chi=deepcopy(q_chi)

    chi[np.arange(chi.shape[0])]=chi[sort_idx]
    q_chi[np.arange(chi.shape[0])]=q_chi[sort_idx]
    chi_spline=splrep(q_chi,chi,k=3)
    def eval_chi(q):
        return splev(q,chi_spline)

    ax=plot(q_chi,chi,'$\chi$',eval_chi)
    ax.set_xlabel(r'$Q$')
    ax.set_ylabel(r'$\chi(Q)$')
    fg=ax.figure
    timestamp=now.strftime('%Y%m%d_%H%M%S')
    fg.savefig(os.path.join('output','chi_'+timestamp))

def get_pol_vs_e(q_energy, energy, celldims, q_chi, chi, E_samples, Ps, debug, method='TNC',close_prev_figs=True):
    """
    Calculates equilibrium polarization value for various electric field values.
    Parameters
    ==========
    q_energy: array(float)
        Collective coordinate values corresponding to data in energy variable.
        Values must be between 0 and 1.
    energy: array(float)
        In hartrees. Bulk energy per unit cell for atomic configurations
        corresponding to q_energy entries.
    celldims: array(float)
        In bohr. Dimensions of unit cell.
    q_chi: array(float)
        Collective coordinate values corresponding to data in chi variable.
        Values must be between 0 and 1.
    chi: array(float)
        In esu. Electronic contribution to dielectric susceptibility for
        configurations along path between ferroelectric configurations.
    E_samples: array(float)
        In kV/cm. Values of electric field at which equilibrium polarisation
        value should be calculated.
    Ps: float
        In uC/cm^2. Spontaneous polarisation of ferroelectric.
    method: string
        Free energy minimization method. See scipy.optimize.minimize
        documentation.
    close_prev_figs: boolean
        If True, previous Matplotlib figures will be closed
    Returns
    =======
    ret: array(float)
        The equilibrium polarizations corresponding to values in E_samples.
    """
    c=2.9979e10
    # au = hartree/bohr^3
    barye_on_au=3.399e-15
    kVpercm_on_statVpercm=1.e5*1.e6/c
    uCpercm2_on_statCpercm2=1e-7*c
    kilopascal_on_au=1e3/(2.942e13)
    global now

    energy=deepcopy(energy)
    e_poly=np.polyfit(q_energy,energy/np.prod(celldims),6)
    e_polyder=np.polyder(e_poly)

    # Sort chi(q) by ascending q.
    sort_idx=np.argsort(q_chi)
    chi=deepcopy(chi)
    q_chi=deepcopy(q_chi)

    chi[np.arange(chi.shape[0])]=chi[sort_idx]
    q_chi[np.arange(chi.shape[0])]=q_chi[sort_idx]
    chi_spline=splrep(q_chi,chi,k=3)

    if close_prev_figs:
        plt.close('all')

    print_fmin_params=False
    def free_energy(q):
        """
        Free energy density in hartrees per bohr according to Garrity eq 1.
        """
        bulk=np.polyval(e_poly,q)
        # uC/cm2 * kV/cm = kPa
        dipole1=-(Ps*(q*2-1)*E*kilopascal_on_au)
        # barye = cgs unit of pressure or energy density
        dipole2=-((0.5*splev(q,chi_spline)*((E*kVpercm_on_statVpercm)**2))*barye_on_au)
        total=bulk+dipole1+dipole2
        if print_fmin_params:
            print("q:{:.5e}; E:{:.5e}; F_bulk:{:.5e}; F_dip1:{:.5e}; F_dip2:{:.5e}; F_tot:{:.5e}".format(float(q),float(E),float(bulk),float(dipole1),float(dipole2),float(total)))
        return total

    def free_energy_derivative(q):
        bulk=np.polyval(e_polyder,q)
        dipole1=-(Ps*2*E*kilopascal_on_au)
        dipole2=-((0.5*spalde(q,chi_spline)*((E*kVpercm_on_statVpercm)**2))*barye_on_au)

        total=bulk+dipole1+dipole2
        if print_fmin_params:
            print("q:{:.5e}; E:{:.5e}; dF_bulk:{:.5e}; dF_dip1:{:.5e}; dF_dip2:{:.5e}; dF_tot:{:.5e}".format(float(q),float(E),float(bulk),float(dipole1),float(dipole2),float(total)))
        return total

    q_start=0.5
    ret=np.zeros((len(E_samples),))
    for i,e in enumerate(E_samples):
        E=e
        res=minimize(free_energy,q_start,method=method,bounds=((-0.5,1.5),))
        if res.success:
            ret[i]=(res.x*2-1.)*Ps
            q_start=res.x
            if debug:
                print('Minimized free energy. P=%f, E=%f. Parameter values:'%(ret[i], E))
                print_fmin_params=debug
            free_energy(res.x)
            print_fmin_params=False
        else:
            ret[i]=math.nan

    return ret

"""
Demonstrates hysteresis_loop() using susceptibility and bulk energy density
calculations of croconic acid (CRCA) by DFT.
"""
if __name__ == "__main__":
    params_obj=yaml.load(open(sys.argv[1]))
    cell_dims=np.array(params_obj['cell_dims'])
    energy_data=np.loadtxt(params_obj['energy_data'])
    chi_data=np.loadtxt(params_obj['chi_data'])
    remnant_polarisation=params_obj['remnant_polarisation']
    Emax=params_obj['Emax']
    Esamples=params_obj['Esamples']
    hysteresis_loop(energy_data[:,0],energy_data[:,1],cell_dims,chi_data[:,0],chi_data[:,1],Emax,Esamples,remnant_polarisation,False)
