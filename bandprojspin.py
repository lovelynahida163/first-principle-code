import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def get_projection_data(atom, axis):
    filename = f"PBAND_{atom}_SOC_S{axis.upper()}.dat"
    if not os.path.exists(filename):
        print(f"Error: Cannot find file {filename}")
        return None
    data = np.loadtxt(filename)
    return data

def main():
    print("--- VASP/VASPKIT Spin Projection Band Tool ---")
    
    all_pbands = glob.glob("PBAND_*_SOC_SX.dat")
    available_atoms = [f.split('_')[1] for f in all_pbands]
    
    if not available_atoms:
        print("Error: No PBAND_..._SOC_SX.dat files found.")
        return

    print(f"Detected atoms: {available_atoms}")
    user_input = input("Enter target atom(s) (e.g., Mn): ")
    target_atoms = user_input.split()

    print("\nSelect Quantization Axis mode:")
    print("[1] Angle in X-Y plane (Degrees)")
    print("[2] Custom Vector (X Y Z)")
    mode = input("Enter choice (1 or 2): ")

    if mode == '1':
        angle_deg = float(input("Enter angle with X-axis (e.g., 60): "))
        theta = np.radians(angle_deg)
        vec = [np.cos(theta), np.sin(theta), 0]
    else:
        vec_input = input("Enter vector (e.g., 0.5 0.866 0): ")
        vec = list(map(float, vec_input.split()))
        vec = np.array(vec) / np.linalg.norm(vec)

    print(f"Processing spin projection along vector: {vec}")

    final_k = None
    final_energy = None
    final_spin_proj = None

    for atom in target_atoms:
        sx_data = get_projection_data(atom, 'X')
        sy_data = get_projection_data(atom, 'Y')
        sz_data = get_projection_data(atom, 'Z')
        
        if sx_data is None: continue

        k_path = sx_data[:, 0]
        energy = sx_data[:, 1]
        
        sx_weight = sx_data[:, 11]
        sy_weight = sy_data[:, 11]
        sz_weight = sz_data[:, 11]

        current_proj = sx_weight * vec[0] + sy_weight * vec[1] + sz_weight * vec[2]

        if final_spin_proj is None:
            final_k, final_energy = k_path, energy
            final_spin_proj = current_proj
        else:
            final_spin_proj += current_proj

    # --- Plotting Section ---
    plt.figure(figsize=(8, 6), dpi=300)

    # Read KLABELS using the provided robust method
    knames = []
    kcoords = []
    if os.path.exists("KLABELS"):
        with open("KLABELS", 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 2 and parts[1][0].isdigit():
                    knames.append(parts[0].replace('GAMMA', r'$\Gamma$'))
                    kcoords.append(float(parts[1]))

    # Scatter plot for fat band with color mapping
    sc = plt.scatter(final_k, final_energy, c=final_spin_proj, s=5, 
                     cmap='RdBu', vmin=-0.5, vmax=0.5, alpha=0.8)
    
    plt.colorbar(sc, label=f'Spin Projection')
    
    # Fermi level
    plt.axhline(0, color='gray', linestyle=':', linewidth=1.2)

    # High symmetry lines and labels
    if kcoords:
        for kc in kcoords:
            plt.axvline(x=kc, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        plt.xticks(kcoords, knames, fontsize=12)
        plt.xlim(0, max(kcoords))
    else:
        plt.xlim(min(final_k), max(final_k))

    plt.ylabel('$E - E_F$ (eV)', fontsize=14)
    plt.ylim([-4, 4]) 
    
    if mode == '1':
        plt.title(f"Spin-Projected Bands: {target_atoms} (Axis: {angle_deg} deg)", fontsize=16)
    else:
        plt.title(f"Spin-Projected Bands: {target_atoms} (Axis: {vec})", fontsize=16)
    
    plt.tight_layout()
    output_fig = f"Spin_Band_{'_'.join(target_atoms)}.png"
    plt.savefig(output_fig)
    print(f"\n--- SUCCESS! Figure saved as: {output_fig} ---")

if __name__ == "__main__":
    main()
