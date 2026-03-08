
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import os

# Use 'Agg' backend for terminal-based HPC environments
matplotlib.use('Agg')

def get_fermi_level(filename="OUTCAR"):
    """Extract Fermi Level from the main OUTCAR file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        ef_match = re.search(r"E-fermi\s*:\s*([\d\.-]+)", content)
        if not ef_match:
            ef_match = re.search(r"Fermi energy:\s*([\d\.-]+)", content)
        return float(ef_match.group(1)) if ef_match else 0.0
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using Ef = 0.0")
        return 0.0

def get_reciprocal_lattice(poscar="POSCAR"):
    """Parse POSCAR to calculate Reciprocal Lattice Vectors (B-matrix)."""
    try:
        with open(poscar, 'r') as f:
            lines = f.readlines()
        scale = float(lines[1].strip())
        a1 = np.array(list(map(float, lines[2].split()))) * scale
        a2 = np.array(list(map(float, lines[3].split()))) * scale
        a3 = np.array(list(map(float, lines[4].split()))) * scale
        volume = np.dot(a1, np.cross(a2, a3))
        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume
        b3 = 2 * np.pi * np.cross(a1, a2) / volume
        return np.array([b1, b2, b3])
    except Exception as e:
        print(f"Error parsing POSCAR: {e}")
        return None

def parse_vasp_independent(filename, b_matrix, ef):
    """
    Extracts K-points and Energies from OUTCAR.band.
    Handles inhomogeneous shapes by filtering out broken k-point blocks.
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return []

    with open(filename, 'r') as f:
        content = f.read()

    # Split by k-point keyword
    k_blocks = content.split("k-point")[1:]
    spin1_kpts, spin1_en = [], []
    spin2_kpts, spin2_en = [], []

    # Robust regex for floats (scientific notation included)
    num_re = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+"

    for block in k_blocks:
        lines = block.strip().split('\n')
        if not lines: continue
        header = lines[0]

        try:
            coords_part = header.split(':')[-1]
            k_coord = [float(x) for x in re.findall(num_re, coords_part)]
            if len(k_coord) != 3: continue
        except: continue

        # Extract energy column: index, energy, occupation
        energies = re.findall(r"^\s*\d+\s+([\d\.-]+)\s+[\d\.-]+", block, re.MULTILINE)
        if not energies: continue
        e_vals = [float(e) - ef for e in energies]

        if "spin component 2" in block:
            spin2_kpts.append(k_coord)
            spin2_en.append(e_vals)
        else:
            spin1_kpts.append(k_coord)
            spin1_en.append(e_vals)

    def build_bands(k_list, e_list):
        if not e_list: return None

        # --- Handle inhomogeneous shapes (e.g., if the last k-point is incomplete) ---
        counts = [len(e) for e in e_list]
        if not counts: return None
        correct_num = max(set(counts), key=counts.count)

        # Filter out k-points that don't match the standard band count
        valid_idx = [i for i, e in enumerate(e_list) if len(e) == correct_num]
        if not valid_idx: return None

        clean_k = np.array([k_list[i] for i in valid_idx])
        clean_e = np.array([e_list[i] for i in valid_idx]) # (kpts, bands)

        # Calculate Cartesian path distances for VASP specifically
        k_cart = np.dot(clean_k, b_matrix)
        diffs = np.diff(k_cart, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        x_axis = np.insert(np.cumsum(dists), 0, 0.0)

        # Format into list of [X, Y] arrays for plotting
        return [np.column_stack((x_axis, clean_e[:, b])) for b in range(clean_e.shape[1])]

    final_spins = []
    for k, e in [(spin1_kpts, spin1_en), (spin2_kpts, spin2_en)]:
        res = build_bands(k, e)
        if res: final_spins.append(res)
    return final_spins

def parse_wannier_dat(filename="wannier90_band.dat", ef=0.0):
    """Read Wannier90 band data (X-axis is already distance)."""
    bands, current_band = [], []
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_band:
                    bands.append(np.array(current_band))
                    current_band = []
                continue
            cols = line.split()
            if len(cols) >= 2:
                current_band.append([float(cols[0]), float(cols[1]) - ef])
    if current_band: bands.append(np.array(current_band))
    return bands

def parse_labels(gnu_file="wannier90_band.gnu"):
    """Extract ticks and labels from the Wannier90 GNU file."""
    ticks, labels = [], []
    try:
        with open(gnu_file, 'r') as f:
            for line in f:
                if "set xtics" in line:
                    matches = re.findall(r'"([^"]+)"\s+([\d\.]+)', line)
                    for m in matches:
                        labels.append(m[0].replace('G', r'$\Gamma$'))
                        ticks.append(float(m[1]))
    except: pass
    return ticks, labels

def main():
    # 1. Physical Setup
    ef = get_fermi_level("OUTCAR")
    b_matrix = get_reciprocal_lattice("POSCAR")

    # 2. Independent Data Retrieval
    # VASP data with its own distance-based X-axis
    vasp_spins = parse_vasp_independent("OUTCAR.band", b_matrix, ef)
    # Wannier90 data with its own X-axis
    wan_bands = parse_wannier_dat("wannier90_band.dat", ef)

    ticks, labels = parse_labels("wannier90_band.gnu")

    # 3. Plotting
    plt.figure(figsize=(8, 10))

    # Plot VASP (Dashed Grey) - Using independent path
    for s_idx, spin_bands in enumerate(vasp_spins):
        for b_idx, band_xy in enumerate(spin_bands):
            lbl = f"VASP" if b_idx == 0 else ""
            plt.plot(band_xy[:, 0], band_xy[:, 1], color='grey', ls='--', lw=1.2, alpha=0.5, label=lbl)

    # Plot Wannier90 (Solid Blue) - Using its own path
    for b_idx, band_xy in enumerate(wan_bands):
        lbl = "Wannier" if b_idx == 0 else ""
        plt.plot(band_xy[:, 0], band_xy[:, 1], color='blue', lw=1.6, alpha=0.8, label=lbl)

    # 4. Polish Plot
    plt.axhline(0, color='red', lw=1.0, ls=':', alpha=0.7) # Fermi level line

    if ticks:
        plt.xticks(ticks, labels, fontsize=12)
        for t in ticks:
            plt.axvline(t, color='black', lw=0.6, alpha=0.3)
        plt.xlim(ticks[0], ticks[-1])

    plt.ylim(-10, 8)
    plt.ylabel(r"$E - E_f$ (eV)", fontsize=14)
    plt.title("Physical Band Alignment (Independent Paths)", fontsize=14)

    # Remove duplicate labels from the legend
    handles, lbls = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig("VASP_Wannier_Comparison.png", dpi=300)
    print("Success: Final plot saved as 'VASP_Wannier_Comparison.png'")

if __name__ == "__main__":
    main()
