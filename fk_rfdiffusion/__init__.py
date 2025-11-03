"""
pepdiff: Feynman-Kac guided RFDiffusion

A wrapper around RFDiffusion that adds Feynman-Kac particle filtering
for optimizing binding affinity and other design objectives.
"""

__version__ = "0.1.0"
__author__ = "Erik Hartman"

# Make main components easily accessible
try:
    from .feynman_kac import create_fk_sampler, FeynmanKacSampler
    __all__ = ['create_fk_sampler', 'FeynmanKacSampler']
except ImportError:
    # Dependencies not available
    __all__ = []
