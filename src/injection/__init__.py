"""Fault injection engine — physics-constrained ramp-mode OBD-II fault injection."""

from .fault_injector import FaultType, InjectionParams, inject_fault, inject_session

__all__ = ["FaultType", "InjectionParams", "inject_fault", "inject_session"]
