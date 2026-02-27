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

    @staticmethod
    def eq101(T, coeffs):
        """Ecuación estándar para Presión de Vapor (VP)"""
        # Form: exp(A + B/T + C*ln(T) + D*T^E)
        A, B, C, D, E, F, G = coeffs
        return np.exp(A + B/T + C*np.log(T) + D*(T**E))

    @staticmethod
    def eq102(T, coeffs):
        """Ecuación Rackett para Densidad Líquida (LDN)"""
        # Form: A / B^(1 + (1 - T/C)^D)
        A, B, C, D, E, F, G = coeffs
        return A / (B**(1 + (1 - T/C)**D))

    @staticmethod
    def eq105(T, coeffs):
        """Ecuación extendida para Densidad Líquida (LDN)"""
        # Form: A / B^(1 + (1 - T/C)^D) - A veces variaciones en exponentes
        A, B, C, D, E, F, G = coeffs
        tau = 1 - (T / C)
        return A / (B**(1 + tau**D))

    @staticmethod
    def eq106(T, coeffs):
        """Ecuación para Entalpía de Vaporización (HVP)"""
        # Form: A * (1-Tr)^(B + C*Tr + D*Tr^2 + E*Tr^3)
        A, B, C, D, E, F, G = coeffs
        Tr = T / C # Donde C suele ser Tc
        tau = 1 - Tr
        exp_term = B + C*Tr + D*(Tr**2) + E*(Tr**3)
        return A * (tau**exp_term)

    @staticmethod
    def eq107(T, coeffs):
        """Ecuación para Capacidad Calorífica (ICP / LCP)"""
        # Form: A + B*T + C*T^2 + D*T^3 + E*T^4
        A, B, C, D, E, F, G = coeffs
        return A + B*T + C*(T**2) + D*(T**3) + E*(T**4)

    @staticmethod
    def eq127(T, coeffs):
        """
        Ecuación de Einstein/Planck (ICP)
        Ideal para Cp de gas ideal sobre rangos térmicos extensos.
        """
        A, B, C, D, E, F, G = coeffs
        
        def einstein_term(temp, coeff_val, char_temp):
            # Evitar división por cero o temperaturas extremadamente bajas
            if temp < 0.1: return 0.0
            x = char_temp / temp
            # Manejo de estabilidad numérica para exp(x)
            if x > 500: return 0.0 
            return coeff_val * (x**2 * np.exp(x) / (np.exp(x) - 1.0)**2)

        term1 = A
        term2 = einstein_term(T, B, C)
        term3 = einstein_term(T, D, E)
        term4 = einstein_term(T, F, G)
        
        return term1 + term2 + term3 + term4

    @classmethod
    def calc(cls, eq_id, T, coeffs):
        """Selector con soporte para Eq. 127"""
        dispatch = {
            101: cls.eq101, 102: cls.eq102, 105: cls.eq105,
            106: cls.eq106, 107: cls.eq107, 127: cls.eq127
        }
        
        func = dispatch.get(int(eq_id))
        if func:
            try:
                # DIPPR usa 0.0 para coeficientes no usados, lo cual es seguro aquí
                return func(T, coeffs)
            except Exception:
                return np.nan
        return np.nan
