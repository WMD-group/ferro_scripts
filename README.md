`hysteresis.py` generates a hysteresis loop of a ferroelectric material according to [Garrity et al 2014](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.112.127601).

Requires [Numpy](http://www.numpy.org/), [Scipy](https://www.scipy.org/), [Matplotlib](https://matplotlib.org/), [PyYAML](https://pyyaml.org/).


To generate hysteresis loop, first, create YAML file with values for

- `cell_dims`: dimensions of unit cell in bohr.
- `energy_data`: path to text file containing space-delimited table. Column 1 contains values of effective coordinate Q (normalised between -1 and 1) expressing an atomic configuration in a 3N-dimensional space between the 2 relaxed ferroelectric configurations. Column 2 contains corresponding values of the energy per unit cell in hartrees.
- `chi_data`: path to text file containing space-delimited table. Column 1 contains values of effective coordinate Q (normalised between -1 and 1) expressing an atomic configuration in a 3N-dimensional space between the 2 relaxed ferroelectric configurations. Column 2 contains corresponding values of the electronic contribution to the linear polarizability (in cgs units).
- `remnant_polarisation`: remnant polarisation of ferroelectric in uC/cm<sup>2.
- `Emax`: maximum external field at which to calculate equilibrium polarization in kV/cm.
- `Esamples`: number of electric field values between -Emax and +Emax at which polarization will be calculated.
- `debug`: int indicating whether contributions to free energy should be printed during minimizations.
 - 0: never output;
 - 1: output after minimization;
 - 2: output at each free energy evaluation during minimization.
See `ferro_scripts/example_params.yaml` for an example.


Then `cd` into the `ferro_scripts` directory and run
`python hysteresis.py <path to params YAML file>`.
