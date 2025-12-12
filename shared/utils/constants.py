"""
ARPES Physics Constants Module

Centralized location for all physical constants used in ARPES data analysis.
Users can easily modify these values for their specific experimental setup.

Physical Background
-------------------
The fundamental ARPES equation relates emission angle to parallel momentum:

    k_parallel (Å⁻¹) = K_FACTOR × √(E_kinetic) × sin(θ)

where K_FACTOR = √(2m_e) / ℏ with proper unit conversions results in
the well-known value of approximately 0.5123 Å⁻¹/√eV.

For perpendicular momentum (kz), the free-electron final state model gives:

    kz (Å⁻¹) = K_FACTOR × √(E_kinetic × cos²(θ) + V0)

where V0 is the inner potential of the material.

Usage
-----
    from shared.utils.constants import K_FACTOR, DEFAULT_V0
    
    # Calculate parallel momentum
    k_parallel = K_FACTOR * np.sqrt(Ek) * np.sin(theta)
"""

# =============================================================================
# Fundamental ARPES Constants
# =============================================================================

# ARPES momentum conversion factor
# k (Å⁻¹) = K_FACTOR * sqrt(Ek (eV)) * sin(theta)
# Derived from: K_FACTOR = sqrt(2 * m_e) / ℏ with proper unit conversions
# This is a well-known value in ARPES physics literature
K_FACTOR: float = 0.5123  # Å⁻¹ / sqrt(eV)


# =============================================================================
# Default Experimental Parameters
# =============================================================================

# Default inner potential (V0) for free-electron final state model
# Used in kz calculations: kz = K_FACTOR * sqrt(Ek * cos²(θ) + V0)
# Typical range: 10-15 eV depending on the material
# Common values: ~10 eV for simple metals, ~12-15 eV for transition metals
DEFAULT_V0: float = 12.57  # eV

# Default work function (φ)
# Used to convert between kinetic and binding energy: E_binding = hν - E_kinetic - φ
# Typical values: 4.0-5.5 eV depending on the analyzer and sample
DEFAULT_WORK_FUNCTION: float = 4.5  # eV


# =============================================================================
# Physical Constants (for reference)
# =============================================================================

# Electron mass in eV/c²
ELECTRON_MASS_EV: float = 511e3  # eV/c²

# Planck constant times speed of light
HC_EV_ANGSTROM: float = 12398.42  # eV·Å


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'K_FACTOR',
    'DEFAULT_V0',
    'DEFAULT_WORK_FUNCTION',
    'ELECTRON_MASS_EV',
    'HC_EV_ANGSTROM',
]
