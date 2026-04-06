# Clinical Lab Tool - Beer-Lambert Law 
Material = input("Name of material: ")

numbers_of_samples = int(input("Numbers of samples: "))
Conc_SD = float(input("Concentration of SD: "))
Ab_SD = float(input("Absorbance of SD: "))

high_limit = float(input("Highest natural result: "))
low_limit = float(input("Lowest natural result: "))

for i in range(numbers_of_samples):
    print(f"\n--- Sample {i+1} ---")
    Ab_S = float(input("Absorbance of sample: "))
    
    conc = (Ab_S / Ab_SD) * Conc_SD
    
    if conc > high_limit:
        print(f"Result: {conc:.2f} (High)")
    elif conc < low_limit:
        print(f"Result: {conc:.2f} (Low)")
    else:
        print(f"Result: {conc:.2f} (Normal)")

print("\nProcessing finished successfully.")