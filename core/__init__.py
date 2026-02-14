"""
Core DSP Module
Contains signal processing logic, I/O functions, and validators.
"""

from .signal_io import (
    ReadSignalFile,
    ReadFreqSignalFile,
    read_samples,
)

from .transformations import (
    SignalComapreAmplitude,
    SignalComaprePhaseShift,
    RoundPhaseShift,
)

from .validators import (
    AddSignalSamplesAreEqual,
    SubSignalSamplesAreEqual,
    MultiplySignalByConst,
    ShiftSignalByConst,
    Folding,
    QuantizationTest1,
    QuantizationTest2,
    Compare_Signals,
    Compare_Correlation_Signals,
)

from .filters import (
    Compare_Filter_Signals,
)

__all__ = [
    # signal_io
    'ReadSignalFile',
    'ReadFreqSignalFile',
    'read_samples',
    # transformations
    'SignalComapreAmplitude',
    'SignalComaprePhaseShift',
    'RoundPhaseShift',
    # validators
    'AddSignalSamplesAreEqual',
    'SubSignalSamplesAreEqual',
    'MultiplySignalByConst',
    'ShiftSignalByConst',
    'Folding',
    'QuantizationTest1',
    'QuantizationTest2',
    'Compare_Signals',
    'Compare_Correlation_Signals',
    # filters
    'Compare_Filter_Signals',
]
