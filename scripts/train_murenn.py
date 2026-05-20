import os 

INVERSE_CLASS_MAP = {
    0: 'fan',
    1: 'pump',
    2: 'slider',
    3: 'valve',
}
INVERSE_SNR_MAP = {
    0: '0dB',
    6: '6dB',
    -6: 'min6dB',
}

module_name = "experiments.murenn_svdd"
script_path = f'/home/cz7tygkr@cnrs.fr/eusipcoxz2026'
project_name = "train_murenn"
save_folder = f'/scratch/nautilus/users/cz7tygkr@cnrs.fr/icasspxz2026/scripts/{project_name}'
os.makedirs(save_folder, exist_ok=True)

# JQT hyperparameters
J = [8, 6]
Q = [8, 1]
T = [32, 3]
J_phi = [8, 6]

# Optimization hyperparameters
dropout_rate = 0.0
lr = 1e-3

# Data folders
snrs = [0, 6, -6]
machine_types = [0, 1, 2, 3]
machine_ids = [0, 2, 4, 6]
split_indices = list(range(5))

# Create folder.
sbatch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), project_name)
os.makedirs(sbatch_dir, exist_ok=True)

experiment_names = []

for snr in [-6, 0, 6]:
    for machine_type in machine_types:
        for machine_id in machine_ids:
            for split_idx in split_indices:
                experiment_name = (
                    f"snr{snr}_machine{machine_type}_"
                    f"id{machine_id}_fold{split_idx}"
                ).replace('.', '_')
                experiment_names.append(experiment_name)
                sbatch_file_path = os.path.join(sbatch_dir, f"{experiment_name}.sbatch")

                cmd_args = [
                    module_name,
                    "--arch LearnableScattering",
                    f"--save_folder {save_folder}/{experiment_name}",
                    f"--J {' '.join(map(str, J))}",
                    f"--Q {' '.join(map(str, Q))}",
                    f"--T {' '.join(map(str, T))}",
                    f"--J_phi {' '.join(map(str, J_phi))}",
                    f"--machine_type {machine_type}",
                    f"--machine_id {machine_id}",
                    f"--snr {snr}",
                    f"--lr {lr}",
                    f"--split_idx {split_idx}",
                ]
                with open(sbatch_file_path, "w") as f:
                    f.write("#!/bin/bash\n\n")

                    f.write(f"#SBATCH --job-name={experiment_name}\n")
                    f.write("#SBATCH --nodes=1\n")
                    f.write("#SBATCH --qos=short\n")
                    f.write("#SBATCH -p gpu\n")
                    f.write("#SBATCH --tasks-per-node=1\n")
                    f.write("#SBATCH --gres=gpu:1\n")
                    f.write("#SBATCH --cpus-per-task=10\n")
                    f.write("#SBATCH --time=2:00:00\n")
                    f.write(f"#SBATCH --output={experiment_name}_%j.out\n")
                    f.write("\n")

                    f.write(
                        'export PATH="/usr/bin:/bin:/usr/sbin:/sbin:'
                        '/usr/local/bin:/usr/local/sbin:$PATH"\n'
                    )
                    f.write("\n")

                    f.write("conda activate murenn_fb\n\n")
                    f.write(f"cd {script_path}\n")
                    f.write(" ".join(["python", "-m"] + cmd_args) + "\n")
                    f.write("\n")


# Generate run.sh
run_path = os.path.join(sbatch_dir, "run.sh")

with open(run_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Submit 5-fold MuReNN SVDD experiments.\n\n")

    for experiment_name in experiment_names:
        file_name = experiment_name + ".sbatch"
        sbatch_path = os.path.join(sbatch_dir, file_name)
        f.write(f"sbatch {sbatch_path}\n")


# Grant permission to execute run.sh
mode = os.stat(run_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(run_path, mode)

print(f"Generated {len(experiment_names)} sbatch files.")
print(f"Run script saved to: {run_path}")