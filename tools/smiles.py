CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}
CHARCANSMILEN = 62
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64
merged_dict = {}
for key, value in CHARCANSMISET.items():
    merged_dict[key] = value
for key, value in CHARISOSMISET.items():
    merged_dict[key] = value
print(merged_dict)
counter = 1
adjusted_dict = {}
for key in merged_dict.keys():
    adjusted_dict[key] = counter
    counter += 1
print(adjusted_dict)