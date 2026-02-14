"""
Signal I/O Module
Contains functions for reading signal files from disk.
"""

def ReadSignalFile(file_name):
    """Read a signal file with indices and samples (4-line header format)."""
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices, expected_samples


def ReadFreqSignalFile(file_name):
    """Read a frequency signal file with amplitude and phase (4-line header format)."""
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                for i in [0, 1]:
                    L[i] = L[i].strip()
                    L[i] = L[i][:-1] if L[i].endswith('f') else L[i]
                V1 = float(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices, expected_samples


def read_samples(file_name):
    """Read a simple samples-only file (no header)."""
    samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        while line:
            V1 = int(line.strip())
            samples.append(V1)
            line = f.readline()
    return samples
