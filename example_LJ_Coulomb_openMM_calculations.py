"""
Provide example manual LJ and Coulomb calculations and set-up for openMM point evaluations of molgri.

Also simple plotting.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import elementary_charge, epsilon_0, N_A
import pandas as pd
import seaborn as sns
from molgri.molecules.writers import PtIOManager
from molgri.scripts.set_up_io import freshly_create_all_folders
import matplotlib.pyplot as plt
import MDAnalysis as mda
import openmm
from openmm.unit import kilojoule, mole, nanometer


#####################################################################################################################
#                      Manual calculation of potentials for two atoms/ions
#####################################################################################################################

def two_atom_Coulomb_calculation(charge1_no_unit: int, charge2_no_unit: int, r_in_A: ArrayLike,
                                 dielectric_constant: float = 1):
    """
    Calculate electrostatic potential between two point particles for an array of different distances r.
    """
    # gives same results as Mia's code
    r_in_m = 1e-10 * r_in_A
    # charges of Cl- and Ca2+
    q1 = charge1_no_unit * elementary_charge  # C
    q2 = charge2_no_unit * elementary_charge  # C
    vacuum_permittivity = epsilon_0  # F m^-1

    potential_in_J = q1*q2/(4*np.pi*vacuum_permittivity*dielectric_constant*r_in_m)
    potential_in_kJ_per_mole = potential_in_J * N_A / 1000
    return potential_in_kJ_per_mole


def combining_rules_LJ_parameters(epsilon1_in_kJ_per_mole: float, epsilon2_in_kJ_per_mole: float,
                                  sigma1_in_A: float, sigma2_in_A: float) -> tuple:
    """
    Uses the typical combining rules to obtain LJ parameters. That means:

    epsilon_ij = sqrt(epsilon_ii * epsilon_jj)
    sigma_ij = (sigma_ii + sigma_jj)/2

    Returns:
        (combined_epsilon, combined sigma) in input units
    """
    combined_epsilon_kj_per_mole = np.sqrt(epsilon1_in_kJ_per_mole * epsilon2_in_kJ_per_mole)
    combined_sigma_A = (sigma1_in_A + sigma2_in_A) / 2
    return combined_epsilon_kj_per_mole, combined_sigma_A


def two_atom_LJ_calculation(r_in_A: ArrayLike, epsilon_in_kJ_per_mole: float, sigma_in_A: float) -> ArrayLike:
    """
    Given combined LJ parameters for a pair of atoms and one or more distances, calculate their LJ potential at
    given distances (r_in_A).

    Returns:
        LJ energy [kJ/mol] in an array of the same shape as r_in_A
    """
    return 4 * epsilon_in_kJ_per_mole*((sigma_in_A/r_in_A)**12 - (sigma_in_A/r_in_A)**6)

#####################################################################################################################
#                         Using openMM for point calculations
#####################################################################################################################


def create_molgri_file(name_molecule_file_1: str, name_molecule_file_2: str, translations_in_A: ArrayLike,
                       num_space_rotations: int = 1, num_body_rotations: int = 1,
                       output_name: str = "current_molgri_trajectory"):
    freshly_create_all_folders()  # don't fail if there are no output folders
    translations_in_nm = str(list(dist/10 for dist in translations_in_A))
    pt_setup = PtIOManager(name_molecule_file_1, name_molecule_file_2, o_grid_name=str(num_space_rotations),
                           b_grid_name=str(num_body_rotations), t_grid_name=translations_in_nm,
                           output_name=output_name)
    pt_setup.construct_pt()
    return pt_setup.get_name()


def point_openmm_calculations_on_pt(molgri_file_name: str, topology_file_path: str):
    path_xtc_file = f"output/pt_files/{molgri_file_name}.xtc"
    path_gro_file = f"output/pt_files/{molgri_file_name}.gro"
    u = mda.Universe(path_gro_file, path_xtc_file)
    frame_num = len(u.trajectory)

    distances_in_A = np.zeros(frame_num)
    potential_energies = np.zeros(frame_num)

    #forcefield = openmm.app.ForceField('amber14/tip3pfb.xml')
    top = openmm.app.GromacsTopFile(topology_file_path, unitCellDimensions=u.dimensions[:3],
                                    includeDir="/home/janjoswig/local/gromacs-2022/share/gromacs/top")
    system = top.createSystem(nonbondedMethod=openmm.app.NoCutoff)

    # integrator will not be used, but formally needs to exist
    integrator = openmm.LangevinMiddleIntegrator(300, 1, 0.004)
    simulation = openmm.app.Simulation(top.topology, system, integrator)

    for i, ts in enumerate(u.trajectory):
        # careful, assumes there are exactly two atoms in the universe!
        distances_in_A[i] = np.linalg.norm(u.atoms.positions[1]-u.atoms.positions[0])
        simulation.context.setPositions(u.atoms.positions / 10)  # openMM expects positions in nm
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        potential_energies[i] = energy.value_in_unit(kilojoule / mole)
    return distances_in_A, potential_energies

#####################################################################################################################
#                                  Plotting tools
#####################################################################################################################


def plot_energies(df, output_name="Ca_Cl_potentials"):
    sns.set_theme(context="talk", style="white")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # LJ plot
    sns.lineplot(x=df["distance [A]"], y=df["LJ energy [kJ/mol]"], label="Calculated LJ", ax=ax)
    # the combined sigma should be the distance at which the LJ potential reaches zero
    ax.vlines(sigma_combined, *ax.get_ylim(), colors="gray", linestyles="--", label=r"$\sigma$")
    # - epsilon is the depth of the potential at minimum
    ax.hlines(-epsilon_combined, *ax.get_xlim(), colors="black", linestyles="--", label=r"$\epsilon$")

    # Coulomb plot
    sns.lineplot(x=df["distance [A]"], y=df["LJ energy [kJ/mol]"], label="Calculated LJ", ax=ax)
    sns.lineplot(x=df["distance [A]"], y=df["Coulomb energy (vacuum) [kJ/mol]"], label="Calculated Coulomb (vacuum)",
                 ax=ax)
    sns.lineplot(x=df["distance [A]"], y=df["Coulomb energy (implicit water) [kJ/mol]"],
                 label="Calculated Coulomb (implicit water)", ax=ax)
    # openMM plot
    sns.scatterplot(x=df["distance [A]"], y=df["openMM energy [kJ/mol]"],
                 label="Full openMM energy (vacuum)", ax=ax, size=1, color="black")
    ax.set_ylabel("Coulomb energy [kJ/mol]")
    ax.set_yscale("symlog")
    plt.tight_layout()
    plt.savefig(f"output/figures/{output_name}")


if __name__ == "__main__":
    # example: Ca(2+) and Cl(-)

    # defining constants for this example
    ca_file = "CA.gro"
    cl_file = "CL.gro"
    topology_file = "input/CA_CL_vacuum.top"
    ca_charge_no_unit = 2
    cl_charge_no_unit = -1
    # from "/home/janjoswig/local/gromacs-2022/share/gromacs/top/amber99sb-ildn.ff/ffnonbonded.itp"
    ca_epsilon_in_kJ_per_mole = 1.92376
    ca_sigma_in_A = 3.0524
    cl_epsilon_in_kJ_per_mole = 0.4184
    cl_sigma_in_A = 4.40104

    # manual calculations
    epsilon_combined, sigma_combined = combining_rules_LJ_parameters(epsilon1_in_kJ_per_mole=ca_epsilon_in_kJ_per_mole,
                                                                     epsilon2_in_kJ_per_mole=cl_epsilon_in_kJ_per_mole,
                                                                     sigma1_in_A=ca_sigma_in_A,
                                                                     sigma2_in_A=cl_sigma_in_A)


    def all_calculations_for_one_set_of_rs(rs_in_A) -> pd.DataFrame:
        """
        To easily compare methods, calculate all of them for a particular set of distances. Relies on parameters
        defined above.
        """
        calculated_LJ_energies_in_kJ_per_mole = two_atom_LJ_calculation(rs_in_A, epsilon_combined, sigma_combined)
        vacuum_Coulomb_energies_in_kJ_per_mole = two_atom_Coulomb_calculation(ca_charge_no_unit, cl_charge_no_unit,
                                                                              rs_in_A)
        water_Coulomb_energies_in_kJ_per_mole = two_atom_Coulomb_calculation(ca_charge_no_unit, cl_charge_no_unit,
                                                                             rs_in_A, 80.2)

        # molgri + openmm calculations
        my_molgri_name = create_molgri_file(ca_file, cl_file, rs_in_A)
        openmm_distances_in_A, openmm_energies_in_kJ_per_mole = point_openmm_calculations_on_pt(my_molgri_name,
                                                                                                topology_file)

        # organising data in a DF
        data = np.array([rs_in_A,
                         calculated_LJ_energies_in_kJ_per_mole,
                         vacuum_Coulomb_energies_in_kJ_per_mole,
                         water_Coulomb_energies_in_kJ_per_mole,
                         openmm_energies_in_kJ_per_mole]).T
        df = pd.DataFrame(data, columns=["distance [A]", "LJ energy [kJ/mol]", "Coulomb energy (vacuum) [kJ/mol]",
                                         "Coulomb energy (implicit water) [kJ/mol]", "openMM energy [kJ/mol]"])
        return df

    interesting_LJ_distances_in_A = np.linspace(0.9*sigma_combined, 2.5*sigma_combined, 100)
    my_df = all_calculations_for_one_set_of_rs(interesting_LJ_distances_in_A)
    plot_energies(my_df, output_name="subset_distances")

    all_distances_in_A = np.linspace(0.1, 10, 1000)
    my_df = all_calculations_for_one_set_of_rs(all_distances_in_A)
    plot_energies(my_df, output_name="many_distances")