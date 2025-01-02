# This module contains frequency mappings for different musical systems used in
# Quantum Harmonics. Each system maps note names to their fundamental frequencies in Hz.
#
# The systems included are:
# - Western 12-tone equal temperament (standard tuning A4=440Hz)
# - Indian Classical (based on Sa=440Hz)
# - Arabic maqam
# - Gamelan (Slendro and Pelog scales)

frequency_systems = {
            'western_12_tone': {
                'C3': 130.81,
                'C#3/Db3': 138.59,
                'D3': 146.83,
                'D#3/Eb3': 155.56,
                'E3': 164.81,
                'F3': 174.61,
                'F#3/Gb3': 185.00,
                'G3': 196.00,
                'G#3/Ab3': 207.65,
                'A3': 220.00,
                'A#3/Bb3': 233.08,
                'B3': 246.94,
                'C4': 261.63,
                'C#4/Db4': 277.18,
                'D4': 293.66,
                'D#4/Eb4': 311.13,
                'E4': 329.63,
                'F4': 349.23,
                'F#4/Gb4': 369.99,
                'G4': 392.00,
                'G#4/Ab4': 415.30,
                'A4': 440.00,
                'A#4/Bb4': 466.16,
                'B4': 493.88
            },
            'indian_classical': {
                'Sa': 440.00,  # Std
                'Re♭': 445.50, # Std
                'Re': 466.16,  # Std, Pyth: 457.14
                'Ga♭': 471.86, # Std
                'Ga': 493.88,  # Std, Pyth: 486.00
                'Ma': 523.25,  # Std, Pyth: 512.00
                'Ma#': 556.88, # Std
                'Pa': 587.33,  # Std, Pyth: 576.00
                'Dha♭': 593.21, # Std
                'Dha': 622.25, # Std, Pyth: 609.52
                'Ni♭': 628.34, # Std
                'Ni': 659.26,  # Std, Pyth: 648.00
            },

            'arabic': {
                'Rast': 269.40,
                'Duka': 302.47,
                'Sika': 339.43,
                'Jaharka': 359.61,
                'Nawa': 403.65,
                'Hussaini': 453.08,
                'Qurdan': 508.52,

            },
            'gamelan': {
                'Slendro1': 508.52,
                'Slendro2': 339.43,
                'Slendro3': 359.61,
                'Slendro4': 403.65,
                'Slendro5': 453.08,
                'Pelog1': 508.52,
                'Pelog2': 339.43
            },
            'pythagorean': {  # Using the Pyth. values from your chart
                'C3': 128.00,
                'C#3': 136.53,
                'D3': 144.00,
                'D#3': 152.38,
                'E3': 162.00,
                'F3': 170.67,
                'E4': 324.00,
                'F4': 341.33,
                'F#3': 181.33,
                'G3': 192.00,
                'G#3': 204.80,
                'A3': 216.00,
                'A#3': 228.57,
                'B3': 243.00
            }
        }