"""
Transformations Module
Contains DFT/IDFT comparison functions for signal transformations.
"""

import math


def SignalComapreAmplitude(SignalInput=[], SignalOutput=[]):
    """Compare amplitude values between two signals for DFT/IDFT testing."""
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i] - SignalOutput[i]) > 0.001:
                return False
            elif SignalInput[i] != SignalOutput[i]:
                return False
        return True


def RoundPhaseShift(P):
    """Normalize phase shift to [0, 2*pi) range."""
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))


def SignalComaprePhaseShift(SignalInput=[], SignalOutput=[]):
    """Compare phase shift values between two signals for DFT testing."""
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            A = round(SignalInput[i])
            B = round(SignalOutput[i])
            if abs(A - B) > 0.0001:
                return False
            elif A != B:
                return False
        return True
