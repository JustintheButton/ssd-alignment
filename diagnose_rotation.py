"""
Diagnostic script to compare rotation analysis with paper's Figure 12.
Run: python diagnose_rotation.py
"""
import numpy as np

# Load the vector data
data = np.load('lora_vectors.npz')
steps = data['steps']
A = data['A']
B = data['B']
gn_steps = data['gn_steps']
grad_norms = data['grad_norms']

print("="*70)
print("DIAGNOSTIC: Comparing your rotation to paper's Figure 12")
print("="*70)

# Paper's formula: local_cos(v_t, k) = cos(v_{t-k} - v_t, v_{t+k} - v_t)
# A peak (toward 0) indicates rotation; -1 means straight line

def local_cos_detailed(V, offset_idx, thresh=0.0035):
    """Compute local cosine with detailed diagnostics."""
    results = []
    for t in range(offset_idx, len(V) - offset_idx):
        before = V[t - offset_idx] - V[t]
        after = V[t + offset_idx] - V[t]
        nb, na = np.linalg.norm(before), np.linalg.norm(after)
        
        if max(nb, na) <= thresh or nb == 0 or na == 0:
            results.append({'step': steps[t], 'cos': np.nan, 'filtered': True, 
                          'nb': nb, 'na': na, 'reason': f'max_norm={max(nb,na):.6f} <= {thresh}'})
        else:
            cos = np.dot(before, after) / (nb * na)
            results.append({'step': steps[t], 'cos': cos, 'filtered': False,
                          'nb': nb, 'na': na, 'reason': ''})
    return results

print("\n1. BASIC STATS")
print("-"*70)
print(f"   Snapshots: {len(steps)}")
print(f"   Step spacing: {steps[1]-steps[0] if len(steps)>1 else 'N/A'}")
print(f"   Step range: {steps[0]} -> {steps[-1]}")
print(f"   B vector dim: {B.shape[1]}")
print(f"   ||B|| range: {np.linalg.norm(B[0]):.6f} -> {np.linalg.norm(B[-1]):.6f}")

print("\n2. THRESHOLD ANALYSIS (κ = 0.0035)")
print("-"*70)
# Check how many points get filtered at different thresholds
for thresh in [0.0035, 0.001, 0.0001, 0.00001]:
    results = local_cos_detailed(B, 1, thresh)
    filtered = sum(1 for r in results if r['filtered'])
    valid = len(results) - filtered
    print(f"   κ={thresh}: {valid}/{len(results)} valid ({100*valid/len(results):.1f}%)")

print("\n3. LOCAL COSINE VALUES (k=5 steps, κ=0.0035)")
print("-"*70)
results = local_cos_detailed(B, 1, thresh=0.0035)
valid_results = [r for r in results if not r['filtered']]

if valid_results:
    cos_values = [r['cos'] for r in valid_results]
    print(f"   Valid points: {len(valid_results)}")
    print(f"   Cos range: {min(cos_values):.4f} to {max(cos_values):.4f}")
    print(f"   Mean cos: {np.mean(cos_values):.4f}")
    
    # Find the peak (maximum cos, indicating rotation)
    peak_idx = np.argmax(cos_values)
    peak = valid_results[peak_idx]
    print(f"\n   PEAK (biggest rotation):")
    print(f"   Step {peak['step']}: cos={peak['cos']:.4f}")
    
    # Compare to paper's expected peak
    print(f"\n   Paper expects: peak around step ~180, cos reaching ~-0.70")
    print(f"   You have:      peak at step {peak['step']}, cos={peak['cos']:.4f}")
else:
    print("   NO VALID POINTS - all filtered by threshold!")
    print("   First 5 filtered points:")
    for r in results[:5]:
        print(f"     Step {r['step']}: {r['reason']}")

print("\n4. EARLY STEP ANALYSIS (where filtering happens)")
print("-"*70)
# Check the first 20 snapshots
print("   Step  ||B||     ||before||  ||after||  Filtered?")
for i in range(min(20, len(B)-2)):
    before = B[i] - B[i+1]
    after = B[i+2] - B[i+1]
    nb, na = np.linalg.norm(before), np.linalg.norm(after)
    filt = "YES" if max(nb, na) <= 0.0035 else "no"
    print(f"   {steps[i+1]:4d}  {np.linalg.norm(B[i+1]):.6f}  {nb:.6f}   {na:.6f}   {filt}")

print("\n5. AROUND EXPECTED ROTATION (steps 150-210)")
print("-"*70)
# Find indices for steps around 180
target_steps = [150, 160, 170, 180, 190, 200, 210]
print("   Step  ||B||     Local cos (k=5)  Local cos (k=10)")
for target in target_steps:
    idx = np.argmin(np.abs(steps - target))
    if idx >= 2 and idx < len(B) - 2:
        # k=5 (offset=1)
        before5 = B[idx-1] - B[idx]
        after5 = B[idx+1] - B[idx]
        nb5, na5 = np.linalg.norm(before5), np.linalg.norm(after5)
        cos5 = np.dot(before5, after5) / (nb5 * na5) if nb5 > 0 and na5 > 0 else np.nan
        
        # k=10 (offset=2)
        if idx >= 2 and idx < len(B) - 2:
            before10 = B[idx-2] - B[idx]
            after10 = B[idx+2] - B[idx]
            nb10, na10 = np.linalg.norm(before10), np.linalg.norm(after10)
            cos10 = np.dot(before10, after10) / (nb10 * na10) if nb10 > 0 and na10 > 0 else np.nan
        else:
            cos10 = np.nan
        
        print(f"   {steps[idx]:4d}  {np.linalg.norm(B[idx]):.6f}  {cos5:+.4f}          {cos10:+.4f}")

print("\n6. DIAGNOSIS")
print("-"*70)

# Check if rotation is present but small
results_k5 = local_cos_detailed(B, 1, thresh=0.0001)  # lower threshold
valid_cos = [r['cos'] for r in results_k5 if not r['filtered'] and not np.isnan(r['cos'])]

if valid_cos:
    max_cos = max(valid_cos)
    min_cos = min(valid_cos)
    range_cos = max_cos - min_cos
    
    if max_cos > -0.80:
        print("   ✓ Sharp rotation detected (cos reaches above -0.80)")
    elif max_cos > -0.90:
        print("   ~ Moderate rotation detected (cos between -0.90 and -0.80)")
    else:
        print("   ✗ Weak/no rotation (cos stays below -0.90)")
        print("     Paper shows cos reaching -0.70 at the rotation peak")
    
    print(f"\n   Your cos range: {min_cos:.4f} to {max_cos:.4f} (span: {range_cos:.4f})")
    print(f"   Paper's range:  -1.00 to -0.70 (span: 0.30)")

print("\n7. POSSIBLE CAUSES")
print("-"*70)
print("""
   If your rotation is weaker than the paper:
   
   a) Different learning dynamics - the phase transition may occur
      differently with your exact setup (tokenization, model version, etc.)
   
   b) The paper's Figure 12 may be from a specific seed/run that showed
      the clearest rotation - not all runs may look identical
   
   c) Section 3.5 vs Footnote 6 HPs - Figure 12's source isn't specified
   
   d) Your training still works (12.5% EM proves it) - the rotation
      metric is diagnostic, not required for EM to emerge
""")
