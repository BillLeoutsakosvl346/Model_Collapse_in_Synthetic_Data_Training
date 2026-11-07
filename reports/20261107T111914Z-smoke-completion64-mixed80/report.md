# FineWeb 80% synthetic pipeline validation

Acceptance `20261107T111914Z-smoke-completion64-mixed80` passed three terminal epochs, 481 regression tests and three host launcher checks. Each small epoch used 256 synthetic and 64 real continuations. These results validate execution and recovery.

The GPU checks reconstructed all mixtures, verified the preceding generator and training checkpoint, persistent optimization state, injected generation/training/coordinator recovery, private cold restoration, reproduced held-out evaluation and watchdog/provider shutdown.

Estimated validation spend was USD 0.16467. Temporary resources were removed after recovery. The full forecast is 2.60 hours and USD 1.51 running cost, plus retained storage. Full launch is authorized after this acceptance.

[Metrics and square graphs](generated/20261107T113544Z-352794a6/report.md). [Training examples](dataset_examples.json).
