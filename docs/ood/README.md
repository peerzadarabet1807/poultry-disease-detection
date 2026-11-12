# Photos that are not poultry droppings

The classifier has no "not poultry droppings" class, so it always answers with Coccidiosis, Healthy, Salmonella or NCD. To see what it does with photos it was never meant for, it was run on 12 openly licensed photos from Wikimedia Commons in three groups: poultry-farm scenes without droppings, droppings of other animals, and food that looks similar. Predictions come from [`scripts/evaluate_ood.py`](../../scripts/evaluate_ood.py) (the Python package, training preprocessing) and are stored in [`predictions.json`](predictions.json).

"Verdict shown" means the app would display a verdict instead of "uncertain" (top probability at least 50% and a lead of at least 90 percentage points over the runner-up).

The last column is what the app does now: the [input check](../../README.md#input-check) runs first, and a photo it turns away gets a notice instead of a verdict.

| # | Photo | Group | Top class | Top probability | Verdict shown | Input check |
|---|---|---|---|---|---|---|
| 1 | [A hen outdoors](hen_free_range.jpg) | farm scene | Healthy | 98.8% | **yes** | turned away |
| 2 | [A tray of eggs](egg_tray.jpg) | farm scene | Coccidiosis | 99.9% | **yes** | turned away |
| 3 | [Wood shavings (litter material)](wood_shavings.jpg) | farm scene | Salmonella | 80.1% | no (uncertain) | turned away |
| 4 | [A heap of used straw bedding in a field](straw_bedding.jpg) | farm scene | NCD | 49.5% | no (uncertain) | turned away |
| 5 | [A bare concrete floor](concrete_floor.jpg) | farm scene | NCD | 57.9% | no (uncertain) | turned away |
| 6 | [Pigeon droppings](pigeon_droppings.jpg) | other droppings | Healthy | 98.8% | **yes** | **passes** |
| 7 | [Wood pigeon droppings](wood_pigeon_droppings.jpg) | other droppings | Coccidiosis | 100.0% | **yes** | turned away |
| 8 | [Cow dung](cow_dung.jpg) | other droppings | Healthy | 99.4% | **yes** | turned away |
| 9 | [Horse manure on a street](horse_manure.jpg) | other droppings | Healthy | 100.0% | **yes** | turned away |
| 10 | [Dog feces on a path](dog_feces.jpg) | other droppings | Healthy | 95.9% | **yes** | turned away |
| 11 | [A bowl of porridge](porridge.jpg) | lookalike food | Healthy | 97.4% | **yes** | turned away |
| 12 | [Oat porridge](oat_porridge.jpg) | lookalike food | Coccidiosis | 61.0% | no (uncertain) | turned away |

**Summary.** A verdict would be shown for 8 of 12 photos; 8 of 12 get a top probability of 90% or more. By group: farm scene, 2 of 5; other droppings, 5 of 5; lookalike food, 1 of 2. The uncertain state catches some of these photos, but it is no substitute for a check that the photo shows poultry droppings at all. The input check turns away 11 of 12.

## The full set: 251 photos for the input check

To build and test the input check, the set was extended to 251 openly licensed photos from Wikimedia Commons (CC0, public domain, CC BY, CC BY-SA; no NC or ND licences). [`build_ood_set.py`](../../scripts/build_ood_set.py) found candidates with fixed search queries per category; the photos were then picked by hand from contact sheets, before any input check was scored ([`selection.json`](selection.json)), and downscaled to 640 px on the long side. They are in [`set/`](set), one folder per category. **Near** photos (169) look like the subject or come from the same setting; **far** photos (82) are unrelated.

[`split_ood.py`](../../scripts/split_ood.py) split the set once, before any tuning, into ood-tune (103) and ood-test (148), stratified by category with a fixed seed ([`split.json`](split.json)); the 12 original photos are all in ood-test. Only ood-tune was used while choosing the input check.

On ood-test, per category: how often the classifier alone would show a verdict (not uncertain), and how often the input check (`tinyclip-8m16-proto-1`, threshold 0.8455) lets the photo through ([`input_check.json`](input_check.json)):

| Category | Near or far | Photos (tune / test) | Verdict shown, classifier alone | Passes the input check |
|---|---|---|---|---|
| cow dung | near | 5 / 8 | 4 of 8 | 0 of 8 |
| dog feces | near | 6 / 9 | 4 of 9 | 0 of 9 |
| eggs | near | 5 / 8 | 6 of 8 | 1 of 8 |
| feathers | near | 6 / 8 | 3 of 8 | 0 of 8 |
| goose duck droppings | near | 6 / 8 | 2 of 8 | 0 of 8 |
| horse manure | near | 5 / 7 | 6 of 7 | 2 of 7 |
| litter bedding | near | 6 / 9 | 4 of 9 | 0 of 9 |
| live poultry | near | 6 / 9 | 3 of 9 | 0 of 9 |
| lookalike food | near | 6 / 10 | 5 of 10 | 0 of 10 |
| other animal droppings | near | 6 / 8 | 3 of 8 | 1 of 8 |
| soil mud | near | 5 / 7 | 4 of 7 | 0 of 7 |
| wild bird droppings | near | 6 / 10 | 5 of 10 | 2 of 10 |
| documents | far | 6 / 8 | 2 of 8 | 0 of 8 |
| everyday objects | far | 6 / 8 | 3 of 8 | 0 of 8 |
| other animals | far | 6 / 8 | 1 of 8 | 0 of 8 |
| people | far | 5 / 7 | 4 of 7 | 0 of 7 |
| scenes | far | 6 / 8 | 3 of 8 | 0 of 8 |
| screenshots | far | 6 / 8 | 0 of 8 | 0 of 8 |
| **All** | | 103 / 148 | 62 of 148 | 6 of 148 |

Rates with 95% intervals, and the false-rejection rates on real droppings photos, are in [`results/input_check/evaluation.md`](../../results/input_check/evaluation.md).

## Credits

The original 12 photos (no 'category' prefix in 'file') were resized by Wikimedia Commons to 1024 px wide with no other changes. The photos under set/ were downscaled as stated in 'changes'. Each photo keeps its own licence; the repository's MIT licence does not apply to them.

| File | Author | License | Source |
|---|---|---|---|
| `hen_free_range.jpg` | safaritravelplus | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Free_Range_Chicken.jpg) |
| `egg_tray.jpg` | safaritravelplus | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:A_Tray_Of_Eggs.jpg) |
| `wood_shavings.jpg` | Intermountain Region US Forest Service | Public domain | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Wood_Shavings_ET5A7918_(30303840940).jpg) |
| `straw_bedding.jpg` | M J Richardson | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Straw-bedding_heap_at_Haugh-head_stables_-_geograph.org.uk_-_5404366.jpg) |
| `concrete_floor.jpg` | W.carter | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Cracked_concrete_floor_in_Brastad.jpg) |
| `pigeon_droppings.jpg` | 01x07x2022000 | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Pigeon_droppings_-or-_Pigeon_poop.JPG) |
| `wood_pigeon_droppings.jpg` | Tony Wills | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:NZ_Wood_pigeon_droppings.jpg) |
| `cow_dung.jpg` | Wilfredor | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Cow_dung_34.jpg) |
| `horse_manure.jpg` | Bonč | Public domain | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Horse_manure_in_Amsterdam.jpg) |
| `dog_feces.jpg` | BrokenSphere | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dog_feces_at_SF_Bay_Trail.JPG) |
| `porridge.jpg` | VirtualSteve | [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Porridge.jpg) |
| `oat_porridge.jpg` | Edithobayaa1 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Oat_porridge_in_Ghana.jpg) |

Credits for all 251 photos: [`CREDITS.md`](CREDITS.md) and [`credits.json`](credits.json). These photos keep their own licenses; the repository's MIT license does not apply to them.
