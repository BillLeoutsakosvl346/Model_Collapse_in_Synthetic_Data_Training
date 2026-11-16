# Whole-epoch removal of real or synthetic contributions

Independent branches start from M2, M5 and M9. Each consumes its original saved epoch targets and follows 79 batch positions at SGD 0.01. Prompt masks, original batch denominators and the 16-example tail remain fixed. Synthetic targets are not regenerated.

The probe figures measure all updates on 128 fixed real-validation pairs. Complete validation/test endpoints are separate. Removing a source preserves its original contribution weight; this is not a separately renormalized 100%-real training run.

| Cycle | Branch | Probe NLL change | Complete test PPL |
| --- | --- | ---: | ---: |
| 3 | both | +0.466145 | 97.944 |
| 3 | real | -0.511662 | 37.317 |
| 3 | synthetic | +0.614938 | 113.164 |
| 6 | both | -0.258004 | 142.696 |
| 6 | real | -1.289128 | 52.257 |
| 6 | synthetic | +0.445227 | 284.060 |
| 10 | both | -0.261655 | 164.412 |
| 10 | real | -1.370186 | 55.652 |
| 10 | synthetic | +0.385075 | 308.995 |

Whole-epoch effects accumulate along different model trajectories and need not add. Cycle 6 was deliberately selected for its historical correction. One seed and these three diagnostic cases do not establish a universal mechanism or permanent stability.

Both-source replay checks must pass before interpreting the ablations. Initializer, targets, masks, RNG state and source selections are pinned; recovery scope is recorded separately.
