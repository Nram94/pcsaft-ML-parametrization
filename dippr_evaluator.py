import warnings

import numpy as np


class DIPPREvaluator:
    """
    Implementación de las ecuaciones estándar de DIPPR.
    Unidades de salida estándar:
    - Presión: Pa
    - Densidad: kmol/m3
    - Entalpía: J/kmol
    - Capacidad Calorífica: J/(kmol·K)
    """

    error_log = []

    @staticmethod
    def eq100(T, coeffs):
        """Ecuación polinómica DIPPR 100."""
        # Form: A + B*T + C*T^2 + D*T^3 + E*T^4
        A, B, C, D, E, F, G = coeffs
        return A + B * T + C * (T**2) + D * (T**3) + E * (T**4)

    @staticmethod
    def eq101(T, coeffs):
        """Ecuación estándar para Presión de Vapor (VP)"""
        # Form: exp(A + B/T + C*ln(T) + D*T^E)
        A, B, C, D, E, F, G = coeffs
        return np.exp(A + B / T + C * np.log(T) + D * (T**E))

    @staticmethod
    def eq102(T, coeffs):
        """Ecuación Rackett para Densidad Líquida (LDN)"""
        # Form: A / B^(1 + (1 - T/C)^D)
        A, B, C, D, E, F, G = coeffs
        return A / (B ** (1 + (1 - T / C) ** D))

    @staticmethod
    def eq105(T, coeffs):
        """Ecuación extendida para Densidad Líquida (LDN)"""
        # Form: A / B^(1 + (1 - T/C)^D) - A veces variaciones en exponentes
        A, B, C, D, E, F, G = coeffs
        tau = 1 - (T / C)
        return A / (B ** (1 + tau**D))

    @staticmethod
    def eq106(T, coeffs):
        """Ecuación para Entalpía de Vaporización (HVP)"""
        # Form: A * (1-Tr)^(B + C*Tr + D*Tr^2 + E*Tr^3)
        A, B, C, D, E, F, G = coeffs
        Tr = T / C  # Donde C suele ser Tc
        tau = 1 - Tr
        exp_term = B + C * Tr + D * (Tr**2) + E * (Tr**3)
        return A * (tau**exp_term)

    @staticmethod
    def eq107(T, coeffs):
        """Ecuación para Capacidad Calorífica Ideal (ICP)"""
        # Form: A + B*((C/T)/sinh(C/T))^2 + D*((E/T)/cosh(E/T))^2
        A, B, C, D, E, F, G = coeffs
        x = C / T
        y = E / T
        return A + B * ((x / np.sinh(x)) ** 2) + D * ((y / np.cosh(y)) ** 2)

    @staticmethod
    def eq114(T, coeffs, critical_temperature=None):
        """Ecuación DIPPR 114 para Cp líquido (forma integrada en τ)."""
        # Form: A^2/τ + B - 2*A*C*τ - A*D*τ^2 - (C^2*τ^3)/3 - (C*D*τ^4)/2 - (D^2*τ^5)/5
        # con τ = 1 - T/Tc y Tc obtenido desde argumento explícito o de los coeficientes.
        A, B, C, D, E, F, G = coeffs
        tc = critical_temperature if critical_temperature is not None else (E if E > 0 else F)
        if tc <= 0:
            raise ValueError("Eq114 requires critical temperature in coefficient E (or F fallback).")
        tau = 1.0 - (T / tc)
        if tau <= 0:
            raise ValueError("Eq114 undefined for tau <= 0.")
        return (
            (A**2) / tau
            + B
            - 2 * A * C * tau
            - A * D * (tau**2)
            - (C**2) * (tau**3) / 3.0
            - C * D * (tau**4) / 2.0
            - (D**2) * (tau**5) / 5.0
        )

    @staticmethod
    def eq119(T, coeffs, critical_temperature=None):
        """Ecuación DIPPR 119 para densidad líquida saturada de agua."""
        # Form: A + B*τ^(1/3) + C*τ^(2/3) + D*τ^(5/3) + E*τ^(16/3) + F*τ^(43/3) + G*τ^(110/3)
        # con τ = 1 - T/Tc y Tc = 647.096 K para esta parametrización.
        A, B, C, D, E, F, G = coeffs
        tc = 647.096 if critical_temperature is None else critical_temperature
        tau = 1.0 - (T / tc)
        if tau < 0:
            raise ValueError("Eq119 undefined for T > Tc.")
        return (
            A
            + B * (tau ** (1.0 / 3.0))
            + C * (tau ** (2.0 / 3.0))
            + D * (tau ** (5.0 / 3.0))
            + E * (tau ** (16.0 / 3.0))
            + F * (tau ** (43.0 / 3.0))
            + G * (tau ** (110.0 / 3.0))
        )

    @staticmethod
    def eq124(T, coeffs, critical_temperature=None):
        """Ecuación DIPPR 124 para Cp líquido en función de temperatura reducida."""
        # Form: A + B/τ + C*τ + D*τ^2 + E*τ^3, con τ = 1 - T/Tc.
        A, B, C, D, E, F, G = coeffs
        tc = critical_temperature if critical_temperature is not None else F
        if tc <= 0:
            raise ValueError("Eq124 requires critical temperature in coefficient F.")
        tau = 1.0 - (T / tc)
        if tau <= 0:
            raise ValueError("Eq124 undefined for tau <= 0.")
        return A + B / tau + C * tau + D * (tau**2) + E * (tau**3)

    @staticmethod
    def eq127(T, coeffs):
        """
        Ecuación de Einstein/Planck (ICP)
        Ideal para Cp de gas ideal sobre rangos térmicos extensos.
        """
        A, B, C, D, E, F, G = coeffs

        def einstein_term(temp, coeff_val, char_temp):
            if temp < 0.1:
                return 0.0
            x = char_temp / temp
            if x > 500:
                return 0.0
            return coeff_val * (x**2 * np.exp(x) / (np.exp(x) - 1.0) ** 2)

        term1 = A
        term2 = einstein_term(T, B, C)
        term3 = einstein_term(T, D, E)
        term4 = einstein_term(T, F, G)

        return term1 + term2 + term3 + term4

    @classmethod
    def get_supported_equation_ids(cls):
        return {
            100,
            101,
            102,
            105,
            106,
            107,
            114,
            119,
            124,
            127,
        }

    @classmethod
    def calc(
        cls,
        eq_id,
        T,
        coeffs,
        chemid=None,
        property_name=None,
        critical_temperature=None,
    ):
        """Selector de ecuación DIPPR con trazabilidad de errores."""
        dispatch = {
            100: cls.eq100,
            101: cls.eq101,
            102: cls.eq102,
            105: cls.eq105,
            106: cls.eq106,
            107: cls.eq107,
            114: cls.eq114,
            119: cls.eq119,
            124: cls.eq124,
            127: cls.eq127,
        }

        eq_id = int(eq_id)
        func = dispatch.get(eq_id)
        trace = (chemid, property_name, eq_id)
        if func is None:
            cls.error_log.append((trace, "unsupported_equation"))
            message = f"Unsupported DIPPR equation encountered: {trace}"
            warnings.warn(message, RuntimeWarning)
            raise NotImplementedError(message)

        try:
            if eq_id in {114, 119, 124}:
                return func(T, coeffs, critical_temperature=critical_temperature)
            return func(T, coeffs)
        except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError) as exc:
            cls.error_log.append((trace, f"{type(exc).__name__}: {exc}"))
            warnings.warn(f"DIPPR evaluation error for {trace}: {exc}", RuntimeWarning)
            return np.nan
