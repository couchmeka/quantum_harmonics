# materials_database.py

"""
Materials Database for Quantum Analysis
Contains physical properties relevant for quantum computations and sound interactions
All properties are in SI units unless otherwise specified
"""

materials_database = {
    # Semiconductors
    "Silicon": {
        "density": 2330,  # kg/m³
        "speed_of_sound": 8433,  # m/s
        "debye_freq": 15.2e12,  # Hz
        "youngs_modulus": 130e9,  # Pa
        "max_temp": 1687,  # K
        "thermal_conductivity": 148,  # W/(m·K)
        "category": "semiconductor",
    },
    "Germanium": {
        "density": 5323,
        "speed_of_sound": 5400,
        "debye_freq": 9.4e12,
        "youngs_modulus": 103e9,
        "max_temp": 1211,
        "thermal_conductivity": 60,
        "category": "semiconductor",
    },
    # Quantum Materials
    "Sapphire": {
        "density": 3980,
        "speed_of_sound": 11100,
        "debye_freq": 29.3e12,
        "youngs_modulus": 400e9,
        "max_temp": 2323,
        "thermal_conductivity": 35,
        "category": "quantum_material",
    },
    "Diamond": {
        "density": 3510,
        "speed_of_sound": 18000,
        "debye_freq": 39.3e12,
        "youngs_modulus": 1220e9,
        "max_temp": 4273,
        "thermal_conductivity": 2200,
        "category": "quantum_material",
    },
    # Superconductors
    "Niobium": {
        "density": 8570,
        "speed_of_sound": 3480,
        "debye_freq": 7.3e12,
        "youngs_modulus": 105e9,
        "max_temp": 2750,
        "critical_temp": 9.3,  # K
        "thermal_conductivity": 54,
        "category": "superconductor",
    },
    "YBCO": {  # Yttrium Barium Copper Oxide
        "density": 6380,
        "speed_of_sound": 4000,
        "debye_freq": 12.5e12,
        "youngs_modulus": 150e9,
        "max_temp": 1300,
        "critical_temp": 93,  # K
        "thermal_conductivity": 2.5,
        "category": "superconductor",
    },
    # Quantum Memory Materials
    "YSO": {  # Yttrium Orthosilicate
        "density": 4440,
        "speed_of_sound": 5000,
        "debye_freq": 14.2e12,
        "youngs_modulus": 135e9,
        "max_temp": 2000,
        "thermal_conductivity": 4.5,
        "category": "quantum_memory",
    },
    "NV_Diamond": {  # Nitrogen-Vacancy Diamond
        "density": 3510,
        "speed_of_sound": 18000,
        "debye_freq": 39.3e12,
        "youngs_modulus": 1220e9,
        "max_temp": 4273,
        "thermal_conductivity": 2200,
        "coherence_time": 1.8e-3,  # seconds
        "category": "quantum_memory",
    },
    # Piezoelectric Materials
    "Quartz": {
        "density": 2650,
        "speed_of_sound": 5760,
        "debye_freq": 8.7e12,
        "youngs_modulus": 76.5e9,
        "max_temp": 1943,
        "thermal_conductivity": 3,
        "piezoelectric_constant": 2.3e-12,  # C/N
        "category": "piezoelectric",
    },
    "LiNbO3": {  # Lithium Niobate
        "density": 4700,
        "speed_of_sound": 7330,
        "debye_freq": 16.4e12,
        "youngs_modulus": 203e9,
        "max_temp": 1530,
        "thermal_conductivity": 5.6,
        "piezoelectric_constant": 6e-11,  # C/N
        "category": "piezoelectric",
    },
}


def get_material_categories():
    """Return a list of all available material categories"""
    return list(set(mat["category"] for mat in materials_database.values()))


def get_materials_by_category(category):
    """Return a list of materials in a specific category"""
    return [
        name
        for name, props in materials_database.items()
        if props["category"] == category
    ]


def get_material_property(material_name, property_name):
    """Get a specific property of a material"""
    if material_name not in materials_database:
        raise KeyError(f"Material '{material_name}' not found in database")
    return materials_database[material_name].get(property_name)


def get_all_materials():
    """Return a list of all available materials"""
    return list(materials_database.keys())


def get_material_properties(material_name):
    """Return all properties of a specific material"""
    if material_name not in materials_database:
        raise KeyError(f"Material '{material_name}' not found in database")
    return materials_database[material_name]
