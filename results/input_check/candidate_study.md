# Input check: candidate study (poultry droppings)

Written by `scripts/gate_report.py` from `candidate_study.json` (`scripts/gate_study.py`) and `browser_bench.json` (`web/scripts/bench-gate.ts`).

**Threshold rule (fixed before any test image was scored):** the highest score threshold that keeps the false-rejection rate (FRR) on the 973 validation images at or below 1%. A photo passes when its score is at or above the threshold. The false-acceptance rate (FAR) is the share of out-of-distribution (OOD) photos that pass. Rates are shown as value (count; 95% Wilson interval).

**Tuning sets** (used to compare and choose): validation images and ood-tune. **Test sets** (reported, not used to choose): 974 internal and 1,201 PCR test images, and ood-test.

## Tuning sets: the basis of the recommendation

| Gate | Variant | AUROC (all OOD) | AUROC (near OOD) | FAR ood-tune, near | FAR ood-tune, far |
|---|---|---|---|---|---|
| resnet50-mahalanobis | feature distance | 0.995 | 0.993 | 20.6% (14/68) | 2.9% (1/35) |
| resnet50-knn10 | feature distance | 0.999 | 0.999 | 1.5% (1/68) | 0.0% (0/35) |
| clip-vit-b32-openai | zero-shot | 0.975 | 0.963 | 33.8% (23/68) | 0.0% (0/35) |
| clip-vit-b32-openai | image prototypes | 1.000 | 1.000 | 0.0% (0/68) | 0.0% (0/35) |
| clip-vit-b32-openai | kNN (k=10) | 1.000 | 1.000 | 0.0% (0/68) | 0.0% (0/35) |
| clip-vit-b32-openai | fused zero-shot + kNN | 1.000 | 1.000 | 0.0% (0/68) | 0.0% (0/35) |
| openclip-vit-b32-laion2b | zero-shot | 0.933 | 0.899 | 35.3% (24/68) | 0.0% (0/35) |
| openclip-vit-b32-laion2b | image prototypes | 1.000 | 0.999 | 2.9% (2/68) | 0.0% (0/35) |
| openclip-vit-b32-laion2b | kNN (k=10) | 1.000 | 1.000 | 0.0% (0/68) | 0.0% (0/35) |
| openclip-vit-b32-laion2b | fused zero-shot + kNN | 0.999 | 0.998 | 2.9% (2/68) | 0.0% (0/35) |
| siglip-base-p16-224 | zero-shot | 0.963 | 0.945 | 45.6% (31/68) | 2.9% (1/35) |
| siglip-base-p16-224 | image prototypes | 0.999 | 0.998 | 4.4% (3/68) | 0.0% (0/35) |
| siglip-base-p16-224 | kNN (k=10) | 0.999 | 0.999 | 1.5% (1/68) | 0.0% (0/35) |
| siglip-base-p16-224 | fused zero-shot + kNN | 0.998 | 0.997 | 8.8% (6/68) | 0.0% (0/35) |
| tinyclip-vit-40m-32-laion400m | zero-shot | 0.938 | 0.906 | 38.2% (26/68) | 0.0% (0/35) |
| tinyclip-vit-40m-32-laion400m | image prototypes | 0.998 | 0.997 | 8.8% (6/68) | 0.0% (0/35) |
| tinyclip-vit-40m-32-laion400m | kNN (k=10) | 0.999 | 0.999 | 4.4% (3/68) | 0.0% (0/35) |
| tinyclip-vit-40m-32-laion400m | fused zero-shot + kNN | 0.998 | 0.997 | 4.4% (3/68) | 0.0% (0/35) |
| tinyclip-vit-39m-16-yfcc15m | zero-shot | 0.928 | 0.891 | 39.7% (27/68) | 0.0% (0/35) |
| tinyclip-vit-39m-16-yfcc15m | image prototypes | 1.000 | 0.999 | 1.5% (1/68) | 0.0% (0/35) |
| tinyclip-vit-39m-16-yfcc15m | kNN (k=10) | 1.000 | 1.000 | 0.0% (0/68) | 0.0% (0/35) |
| tinyclip-vit-39m-16-yfcc15m | fused zero-shot + kNN | 0.999 | 0.999 | 1.5% (1/68) | 0.0% (0/35) |
| tinyclip-vit-8m-16-yfcc15m | zero-shot | 0.889 | 0.832 | 51.5% (35/68) | 0.0% (0/35) |
| tinyclip-vit-8m-16-yfcc15m | image prototypes | 0.997 | 0.996 | 5.9% (4/68) | 0.0% (0/35) |
| tinyclip-vit-8m-16-yfcc15m | kNN (k=10) | 0.999 | 0.998 | 4.4% (3/68) | 0.0% (0/35) |
| tinyclip-vit-8m-16-yfcc15m | fused zero-shot + kNN | 0.996 | 0.993 | 7.4% (5/68) | 0.0% (0/35) |

## Test sets (reported only)

| Gate | Variant | FRR internal test | FRR PCR test | FAR ood-test, near | FAR ood-test, far |
|---|---|---|---|---|---|
| resnet50-mahalanobis | feature distance | 1.0% (10/974) | 8.8% (106/1201) | 18.8% (19/101) | 4.3% (2/47) |
| resnet50-knn10 | feature distance | 2.3% (22/974) | 29.6% (356/1201) | 0.0% (0/101) | 0.0% (0/47) |
| clip-vit-b32-openai | zero-shot | 1.0% (10/974) | 3.2% (39/1201) | 31.7% (32/101) | 0.0% (0/47) |
| clip-vit-b32-openai | image prototypes | 1.6% (16/974) | 6.9% (83/1201) | 2.0% (2/101) | 0.0% (0/47) |
| clip-vit-b32-openai | kNN (k=10) | 1.1% (11/974) | 11.9% (143/1201) | 0.0% (0/101) | 0.0% (0/47) |
| clip-vit-b32-openai | fused zero-shot + kNN | 1.0% (10/974) | 8.2% (98/1201) | 2.0% (2/101) | 0.0% (0/47) |
| openclip-vit-b32-laion2b | zero-shot | 1.0% (10/974) | 1.2% (15/1201) | 38.6% (39/101) | 0.0% (0/47) |
| openclip-vit-b32-laion2b | image prototypes | 0.8% (8/974) | 4.5% (54/1201) | 1.0% (1/101) | 0.0% (0/47) |
| openclip-vit-b32-laion2b | kNN (k=10) | 0.9% (9/974) | 7.2% (86/1201) | 1.0% (1/101) | 0.0% (0/47) |
| openclip-vit-b32-laion2b | fused zero-shot + kNN | 1.2% (12/974) | 3.9% (47/1201) | 5.0% (5/101) | 0.0% (0/47) |
| siglip-base-p16-224 | zero-shot | 0.9% (9/974) | 1.7% (21/1201) | 31.7% (32/101) | 0.0% (0/47) |
| siglip-base-p16-224 | image prototypes | 0.8% (8/974) | 3.2% (38/1201) | 9.9% (10/101) | 0.0% (0/47) |
| siglip-base-p16-224 | kNN (k=10) | 1.0% (10/974) | 6.7% (81/1201) | 6.9% (7/101) | 0.0% (0/47) |
| siglip-base-p16-224 | fused zero-shot + kNN | 0.5% (5/974) | 4.0% (48/1201) | 7.9% (8/101) | 0.0% (0/47) |
| tinyclip-vit-40m-32-laion400m | zero-shot | 1.1% (11/974) | 3.2% (39/1201) | 39.6% (40/101) | 4.3% (2/47) |
| tinyclip-vit-40m-32-laion400m | image prototypes | 1.2% (12/974) | 1.3% (16/1201) | 5.0% (5/101) | 0.0% (0/47) |
| tinyclip-vit-40m-32-laion400m | kNN (k=10) | 0.9% (9/974) | 3.2% (38/1201) | 2.0% (2/101) | 0.0% (0/47) |
| tinyclip-vit-40m-32-laion400m | fused zero-shot + kNN | 0.9% (9/974) | 5.0% (60/1201) | 3.0% (3/101) | 0.0% (0/47) |
| tinyclip-vit-39m-16-yfcc15m | zero-shot | 0.3% (3/974) | 1.2% (14/1201) | 38.6% (39/101) | 0.0% (0/47) |
| tinyclip-vit-39m-16-yfcc15m | image prototypes | 1.6% (16/974) | 5.8% (70/1201) | 1.0% (1/101) | 0.0% (0/47) |
| tinyclip-vit-39m-16-yfcc15m | kNN (k=10) | 1.4% (14/974) | 10.2% (122/1201) | 0.0% (0/101) | 0.0% (0/47) |
| tinyclip-vit-39m-16-yfcc15m | fused zero-shot + kNN | 0.7% (7/974) | 5.1% (61/1201) | 3.0% (3/101) | 0.0% (0/47) |
| tinyclip-vit-8m-16-yfcc15m | zero-shot | 0.3% (3/974) | 1.1% (13/1201) | 43.6% (44/101) | 0.0% (0/47) |
| tinyclip-vit-8m-16-yfcc15m | image prototypes | 0.8% (8/974) | 1.7% (21/1201) | 5.9% (6/101) | 0.0% (0/47) |
| tinyclip-vit-8m-16-yfcc15m | kNN (k=10) | 1.4% (14/974) | 4.6% (55/1201) | 3.0% (3/101) | 0.0% (0/47) |
| tinyclip-vit-8m-16-yfcc15m | fused zero-shot + kNN | 1.3% (13/974) | 4.6% (55/1201) | 4.0% (4/101) | 0.0% (0/47) |

## Licence, size and speed

Download sizes are the files the browser would fetch. Python latency: ONNX Runtime 1.20.1 on this machine's CPU, one image, one thread. Browser latency: median of 20 runs after warm-up; desktop GPU is headed Chromium on this machine's integrated GPU; phone is Pixel 7 emulation with 4x CPU throttling.

| Gate | Licence | Image-encoder parameters | ONNX fp32 / fp16 / int8 | TF.js float16 / uint8 | Python CPU, 1 thread (fp32) | Browser, desktop GPU (TF.js WebGL) | Browser, desktop (TF.js WASM) | Browser, desktop (ORT WASM, int8) | Browser, phone (TF.js WASM) | Browser, phone (ORT WASM, int8) |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet50-mahalanobis | the app's own model (MIT); no extra download | (main model) | none | none | n/a | n/a | n/a | n/a | n/a | n/a |
| resnet50-knn10 | the app's own model (MIT); no extra download | (main model) | none | none | n/a | n/a | n/a | n/a | n/a | n/a |
| clip-vit-b32-openai | MIT (github.com/openai/CLIP LICENSE); model card: any deployed use is out of scope | 87.5 M | 335.3 MiB / 167.8 MiB / 91.3 MiB | 168.0 MiB / 84.2 MiB | 76 ms | 132 ms | 267 ms | 122 ms | 1218 ms | 534 ms |
| openclip-vit-b32-laion2b | MIT (model card); model card: any deployed use is out of scope | 87.8 M | 335.5 MiB / 168.1 MiB / 91.6 MiB | not converted / not converted | 78 ms | not measured | not measured | 123 ms | not measured | 607 ms |
| siglip-base-p16-224 | Apache-2.0 (model card) | 92.9 M | 354.6 MiB / 177.5 MiB / 96.6 MiB | not converted / not converted | 285 ms | not measured | not measured | 481 ms | not measured | 2245 ms |
| tinyclip-vit-40m-32-laion400m | MIT (model card; microsoft/Cream TinyCLIP/LICENSE) | 39.4 M | 151.6 MiB / 76.0 MiB / 43.0 MiB | 76.1 MiB / 38.3 MiB | 38 ms | 116 ms | 133 ms | 62 ms | 625 ms | 307 ms |
| tinyclip-vit-39m-16-yfcc15m | MIT (model card; microsoft/Cream TinyCLIP/LICENSE) | 38.3 M | 147.4 MiB / 73.9 MiB / 38.8 MiB | 74.0 MiB / 37.2 MiB | 118 ms | 226 ms | 673 ms | 231 ms | 3173 ms | 1115 ms |
| tinyclip-vit-8m-16-yfcc15m | MIT (model card; microsoft/Cream TinyCLIP/LICENSE) | 8.1 M | 31.7 MiB / 16.0 MiB / 9.0 MiB | 16.1 MiB / 8.2 MiB | 29 ms | 68 ms | 212 ms | 61 ms | 919 ms | 288 ms |

ONNX Runtime Web adds its own WebAssembly binary: onnxruntime-web 1.20.1 ort-wasm-simd-threaded.wasm, 10.7 MiB (2.9 MiB gzipped). TF.js is already loaded by the app.

## Per-category false acceptance on ood-tune

| Gate | Variant | cow_dung | documents | dog_feces | eggs | everyday_objects | feathers | goose_duck_droppings | horse_manure | litter_bedding | live_poultry | lookalike_food | other_animal_droppings | other_animals | people | scenes | screenshots | soil_mud | wild_bird_droppings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50-mahalanobis | feature distance | 2/5 | 0/6 | 2/6 | 0/5 | 0/6 | 1/6 | 3/6 | 1/5 | 0/6 | 1/6 | 0/6 | 1/6 | 1/6 | 0/5 | 0/6 | 0/6 | 3/5 | 0/6 |
| resnet50-knn10 | feature distance | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| clip-vit-b32-openai | zero-shot | 2/5 | 0/6 | 4/6 | 0/5 | 0/6 | 2/6 | 3/6 | 0/5 | 1/6 | 0/6 | 1/6 | 5/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 4/6 |
| clip-vit-b32-openai | image prototypes | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| clip-vit-b32-openai | kNN (k=10) | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| clip-vit-b32-openai | fused zero-shot + kNN | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| openclip-vit-b32-laion2b | zero-shot | 4/5 | 0/6 | 2/6 | 0/5 | 0/6 | 1/6 | 2/6 | 2/5 | 2/6 | 1/6 | 0/6 | 6/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 3/6 |
| openclip-vit-b32-laion2b | image prototypes | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/5 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 0/6 |
| openclip-vit-b32-laion2b | kNN (k=10) | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| openclip-vit-b32-laion2b | fused zero-shot + kNN | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/5 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 0/6 |
| siglip-base-p16-224 | zero-shot | 3/5 | 0/6 | 5/6 | 0/5 | 0/6 | 1/6 | 3/6 | 2/5 | 3/6 | 2/6 | 0/6 | 5/6 | 0/6 | 1/5 | 0/6 | 0/6 | 1/5 | 6/6 |
| siglip-base-p16-224 | image prototypes | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 1/6 |
| siglip-base-p16-224 | kNN (k=10) | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| siglip-base-p16-224 | fused zero-shot + kNN | 1/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 1/6 | 0/6 | 0/6 | 1/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 1/6 |
| tinyclip-vit-40m-32-laion400m | zero-shot | 3/5 | 0/6 | 5/6 | 0/5 | 0/6 | 2/6 | 3/6 | 0/5 | 2/6 | 1/6 | 0/6 | 4/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 5/6 |
| tinyclip-vit-40m-32-laion400m | image prototypes | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 1/6 | 0/6 | 0/6 | 2/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 1/6 |
| tinyclip-vit-40m-32-laion400m | kNN (k=10) | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 0/6 | 0/6 | 0/6 | 1/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 0/6 |
| tinyclip-vit-40m-32-laion400m | fused zero-shot + kNN | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 1/6 | 0/6 | 0/6 | 1/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| tinyclip-vit-39m-16-yfcc15m | zero-shot | 3/5 | 0/6 | 4/6 | 1/5 | 0/6 | 1/6 | 6/6 | 1/5 | 2/6 | 0/6 | 0/6 | 6/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 2/6 |
| tinyclip-vit-39m-16-yfcc15m | image prototypes | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| tinyclip-vit-39m-16-yfcc15m | kNN (k=10) | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| tinyclip-vit-39m-16-yfcc15m | fused zero-shot + kNN | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |
| tinyclip-vit-8m-16-yfcc15m | zero-shot | 3/5 | 0/6 | 6/6 | 2/5 | 0/6 | 2/6 | 6/6 | 3/5 | 2/6 | 0/6 | 1/6 | 6/6 | 0/6 | 0/5 | 0/6 | 0/6 | 2/5 | 2/6 |
| tinyclip-vit-8m-16-yfcc15m | image prototypes | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 0/6 | 0/6 | 0/6 | 2/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 0/6 |
| tinyclip-vit-8m-16-yfcc15m | kNN (k=10) | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 0/6 | 0/6 | 0/6 | 1/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/5 | 0/6 |
| tinyclip-vit-8m-16-yfcc15m | fused zero-shot + kNN | 1/5 | 0/6 | 0/6 | 0/5 | 0/6 | 0/6 | 1/6 | 0/5 | 1/6 | 0/6 | 0/6 | 2/6 | 0/6 | 0/5 | 0/6 | 0/6 | 0/5 | 0/6 |

Figures: `figures/roc_tune.png` (tuning sets) and `figures/roc_test.png` (test sets).
