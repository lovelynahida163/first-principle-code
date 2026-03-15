import matplotlib.pyplot as plt
import numpy as np

input_file = 'BAND.dat'
label_file = 'KLABELS'
output_image = 'Band_Structure.png'
energy_range = [-4, 4]  

def plot_vasp_band():
    knames = []
    kcoords = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[1][0].isdigit():
                knames.append(parts[0].replace('GAMMA', r'$\Gamma$'))
                kcoords.append(float(parts[1]))

    data = np.loadtxt(input_file)
    k_path = data[:, 0]
    spin_up = data[:, 1]
    spin_down = data[:, 2]

    num_kpts = len(np.unique(k_path))
    num_bands = len(data) // num_kpts

    plt.figure(figsize=(8, 6), dpi=300)
    
    for i in range(num_bands):
        start = i * num_kpts
        end = (i + 1) * num_kpts
        
        line1, = plt.plot(k_path[start:end], spin_up[start:end], 
                         color='#1f77b4', linewidth=1.5, alpha=0.8)
        line2, = plt.plot(k_path[start:end], spin_down[start:end], 
                         color='#ff7f0e', linewidth=1.0, linestyle='--', alpha=0.7)


    for kc in kcoords:
        plt.axvline(x=kc, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1.2)

    plt.xticks(kcoords, knames, fontsize=12)
    plt.ylabel('$E - E_F$ (eV)', fontsize=14)
    plt.ylim(energy_range)
    plt.xlim(0, max(kcoords))
plt.xlim ( 0 , max ( kcoords ) )  

    plt.title('Band Structure (AFM)', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(output_image)
    print(f"Success! Plot saved as {output_image}")
    plt.show()

if __name__ == "__main__":
    plot_vasp_band()
