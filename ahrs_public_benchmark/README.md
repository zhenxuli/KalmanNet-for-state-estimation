# Public-data AI-AHRS benchmark

This experiment evaluates six-axis and nine-axis attitude estimators separately on held-out trajectories from the BROAD dataset.

## Compared methods

- Gyro-only
- Madgwick
- Mahony
- VQF
- Fixed-covariance MEKF
- Norm-adaptive MEKF
- TrustNet-MEKF: learns observation uncertainty only
- RefNet-MEKF: learns clean reference directions only
- ProbRefNet-MEKF: jointly learns reference directions and uncertainty
- Oracle-reference-MEKF: non-deployable upper bound

## Protocol

- BROAD is split by complete trajectory, never by shuffled windows.
- Six-axis evaluation ranks inclination error and relative rotation; yaw is only initial-heading aligned.
- Nine-axis evaluation ranks total, heading, and inclination errors.
- All methods receive the same first ground-truth quaternion to isolate filtering robustness from initial alignment.
- The source supports multi-seed neural ensembles; the CI screening run uses a smaller seed/epoch budget before statistical re-checks.

## Run

```bash
python -m ahrs_public_benchmark.run_benchmark \
  --work-dir .benchmark_work \
  --output-dir benchmark_results \
  --target-hz 50 \
  --window 64 \
  --epochs 6 \
  --seeds 0 1 2
```
