
# OSTEOGENIC PROGRESSION MODEL

---

## 1. Fixed Parameters (Not Fitted)

### Protocol Inputs
- **t_osteo** : osteogenic induction time  
- **Δt_med**  : medium refresh interval  
- **Ethanol carrier**: 0.10 % (v/v) in all conditions  

### Ascorbic Acid (AA)
- **A0** = 0.284 mM  
- **Half-life τ** ≈ 6 h  
- **k_A** = ln(2)/τ ≈ 0.11 h⁻¹  

### AA Collagen Gate
- **K_A^C** (fixed)

### Mineralization Time Gate
- **τ_H** = 14 days  
- **n** = 4  

### Dose Definitions
- **Dex Low (D_L)** = 10 nM  
- **Dex High (D_H)** = 100 nM  
- **MK-4 Low (K_L)** = 0.1 µM  
- **MK-4 High (K_H)** = 10 µM  

---

## 2. Free Parameters (11 Total)

### ALP (4)
- r_b^P  
- r_o^P  
- κ_D^P  
- κ_K^P  

### Collagen (4)
- r_b^C  
- r_o^C  
- κ_D^C  
- κ_K^C  

### Mineral (3)
- r_o^H  
- κ_D^H  
- κ_K^H  

**Total free parameters = 11**

---

## 3. Experimental Conditions (C1–C8)

### C1: Basal
AA=0 | β-GP=0 | Dex=0 | MK-4=0  

### C2: AA + β-GP
AA=0.284 mM | β-GP=10 mM | Dex=0 | MK-4=0  

### C3: AA + β-GP + D_L + K_L
AA=0.284 mM | β-GP=10 mM | Dex=10 nM | MK-4=0.1 µM  

### C4: AA + β-GP + D_L + K_H
AA=0.284 mM | β-GP=10 mM | Dex=10 nM | MK-4=10 µM  

### C5: AA + β-GP + D_H + K_L
AA=0.284 mM | β-GP=10 mM | Dex=100 nM | MK-4=0.1 µM  

### C6: AA + β-GP + D_H + K_H
AA=0.284 mM | β-GP=10 mM | Dex=100 nM | MK-4=10 µM  

### C7: AA + β-GP + D_H + K_0
AA=0.284 mM | β-GP=10 mM | Dex=100 nM | MK-4=0  

### C8: AA + β-GP + D_0 + K_H
AA=0.284 mM | β-GP=10 mM | Dex=0 | MK-4=10 µM  

---

## 4. Replicate Structure

Per condition:
- **ALP**: 3 replicates  
- **Collagen**: 5 replicates  
- **ARS**: 9 replicates  

Total conditions: 8  

