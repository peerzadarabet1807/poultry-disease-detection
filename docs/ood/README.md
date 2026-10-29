# Photos that are not poultry droppings

The classifier has no "not poultry droppings" class, so it always answers with Coccidiosis, Healthy, Salmonella or NCD. To see what it does with photos it was never meant for, it was run on 12 openly licensed photos from Wikimedia Commons in three groups: poultry-farm scenes without droppings, droppings of other animals, and food that looks similar. Predictions come from [`scripts/evaluate_ood.py`](../../scripts/evaluate_ood.py) (the Python package, training preprocessing) and are stored in [`predictions.json`](predictions.json).

"Verdict shown" means the app would display a verdict instead of "uncertain" (top probability at least 50% and a lead of at least 90 percentage points over the runner-up).

| # | Photo | Group | Top class | Top probability | Verdict shown |
|---|---|---|---|---|---|
| 1 | [A hen outdoors](hen_free_range.jpg) | farm scene | Healthy | 98.8% | **yes** |
| 2 | [A tray of eggs](egg_tray.jpg) | farm scene | Coccidiosis | 99.9% | **yes** |
| 3 | [Wood shavings (litter material)](wood_shavings.jpg) | farm scene | Salmonella | 80.1% | no (uncertain) |
| 4 | [A heap of used straw bedding in a field](straw_bedding.jpg) | farm scene | NCD | 49.5% | no (uncertain) |
| 5 | [A bare concrete floor](concrete_floor.jpg) | farm scene | NCD | 57.9% | no (uncertain) |
| 6 | [Pigeon droppings](pigeon_droppings.jpg) | other droppings | Healthy | 98.8% | **yes** |
| 7 | [Wood pigeon droppings](wood_pigeon_droppings.jpg) | other droppings | Coccidiosis | 100.0% | **yes** |
| 8 | [Cow dung](cow_dung.jpg) | other droppings | Healthy | 99.4% | **yes** |
| 9 | [Horse manure on a street](horse_manure.jpg) | other droppings | Healthy | 100.0% | **yes** |
| 10 | [Dog feces on a path](dog_feces.jpg) | other droppings | Healthy | 95.9% | **yes** |
| 11 | [A bowl of porridge](porridge.jpg) | lookalike food | Healthy | 97.4% | **yes** |
| 12 | [Oat porridge](oat_porridge.jpg) | lookalike food | Coccidiosis | 61.0% | no (uncertain) |

**Summary.** A verdict would be shown for 8 of 12 photos; 8 of 12 get a top probability of 90% or more. By group: farm scene, 2 of 5; other droppings, 5 of 5; lookalike food, 1 of 2. The uncertain state catches some of these photos, but it is no substitute for a check that the photo shows poultry droppings at all; only the person taking the photo can make that check.

## Credits

Resized by Wikimedia Commons to 1024 px wide; no other changes. Each photo keeps its own license.

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

These photos keep their own licenses; the repository's MIT license does not apply to them.
