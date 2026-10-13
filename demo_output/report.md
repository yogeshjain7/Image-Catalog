# Demo report

## Scenario 1 - successful detection (names, colours, confidence, boxes)

**apple** - 400x400, status `detected`, detector `classical-cv-v1`

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 1 | apple | red | 0.85 | 90,76,221,250 | auto |

**orange** - 400x400, status `detected`, detector `classical-cv-v1`

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 2 | orange | orange | 0.85 | 95,92,211,214 | auto |

**boy** - 300x400, status `detected`, detector `classical-cv-v1`

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 3 | boy | green | 0.78 | 65,12,170,371 | auto |

> clothing colours: ['green', 'blue'], hair: brown

**girl** - 300x400, status `detected`, detector `classical-cv-v1`

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 4 | girl | purple | 0.78 | 56,12,189,368 | auto |

> clothing colours: ['purple'], hair: brown

**shapes** - 860x260, status `detected`, detector `classical-cv-v1`

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 5 | circle | blue | 0.88 | 30,50,161,161 | auto |
| 6 | square | red | 0.95 | 239,50,141,139 | auto |
| 7 | triangle | green | 0.91 | 433,58,156,148 | auto |
| 8 | rectangle | orange | 0.95 | 649,79,181,101 | auto |

**colors** - 750x180, status `detected`, detector `classical-cv-v1`

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 9 | red swatch | red | 0.95 | 0,0,150,179 | auto |
| 10 | blue swatch | blue | 0.95 | 150,0,150,179 | auto |
| 11 | green swatch | green | 0.95 | 300,0,150,179 | auto |
| 12 | yellow swatch | yellow | 0.95 | 450,0,150,179 | auto |
| 13 | purple swatch | purple | 0.95 | 600,0,150,179 | auto |

**apple_orange** - 760x380, status `detected`, detector `classical-cv-v1`

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 14 | apple | red | 0.85 | 84,71,213,240 | auto |
| 15 | orange | orange | 0.85 | 460,91,201,205 | auto |

**overlapping_shapes** - 520x400, status `detected`, detector `classical-cv-v1`

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 16 | circle | red | 0.81 | 70,80,208,241 | auto |
| 17 | square | blue | 0.87 | 240,119,160,161 | auto |
| 18 | triangle | green | 0.91 | 331,192,138,149 | auto |

## Scenario 7a - overlapping objects

`overlapping_shapes` above: the circle is partly hidden by the square, the square by the triangle. Boxes overlap; the hidden circle is recognised from its *visible arc* (`occluded: true`).

## Scenario 2 - failed / missing detection, then manual add

**blank** -> status `no_objects`, notes ['No detectable objects found.']

**noise_scene** -> status `no_objects`, notes ['Background is not uniform; using GrabCut fallback (low confidence).', 'No detectable objects found.']

User draws a box on the blank image:

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 19 | pencil | gray | 1.00 | 60,50,140,90 | manual |

## Scenario 3 - edit label / colour / confidence / bounding box

**busy_background_apple** (gradient wall). The detector sees a red circle, not an apple (no leaf/stem to go on):

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 20 | circle | red | 0.88 | 150,140,111,111 | auto |

User corrects label, confidence and box. Source becomes `edited`; the detector's answer is kept for audit:

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 20 | apple | red | 0.95 | 148,138,116,116 | edited |

Detector originally said: `{'label': 'circle', 'color': 'red', 'confidence': 0.88, 'x': 150, 'y': 140, 'w': 111, 'h': 111, 'attributes': {'shape': 'circle'}}`; fields edited: `['confidence', 'h', 'label', 'w', 'x', 'y']`

## Scenario 4 - corrections persist until detection is explicitly re-run

Edited the circle, added a ruler, deleted the square. Re-reading the image does NOT re-detect:

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 21 | ruler |  | 1.00 | 5,5,40,12 | manual |
| 5 | blue ball | blue | 0.88 | 30,50,161,161 | edited |
| 7 | triangle | green | 0.91 | 433,58,156,148 | auto |
| 8 | rectangle | orange | 0.95 | 649,79,181,101 | auto |

Re-run with **keep my edits** (added 2, kept 2, skipped 2) - my work survives, deleted square is not resurrected:

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 21 | ruler |  | 1.00 | 5,5,40,12 | manual |
| 5 | blue ball | blue | 0.88 | 30,50,161,161 | edited |
| 22 | triangle | green | 0.91 | 433,58,156,148 | auto |
| 23 | rectangle | orange | 0.95 | 649,79,181,101 | auto |

Re-run with **replace everything** - back to pure detector output:

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 24 | circle | blue | 0.88 | 30,50,161,161 | auto |
| 25 | square | red | 0.95 | 239,50,141,139 | auto |
| 26 | triangle | green | 0.91 | 433,58,156,148 | auto |
| 27 | rectangle | orange | 0.95 | 649,79,181,101 | auto |

## Scenario 2b - detector outage must not destroy corrections

status `failed`, error `ConnectionError: detection service timed out`; objects kept: ['apple', 'stem']

## Scenario 5 - composite from the existing `apple` and `orange` assets

Saved as new catalog entry `composite_apple_orange` (id 12, kind `composite`), layout `row`.

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 29 | apple | red | 1.00 | 229,173,260,293 | composed |
| 30 | orange | orange | 1.00 | 521,189,274,278 | composed |

Provenance: apple[90, 76, 221, 250] -> [229, 173, 260, 293], orange[95, 92, 211, 214] -> [521, 189, 274, 278]

## Scenario 6 - composite with a boy, a girl and colored shapes

| id | label | color | conf | bbox (x,y,w,h) | source |
|---|---|---|---|---|---|
| 31 | boy | green | 1.00 | 44,141,166,358 | composed |
| 32 | girl | purple | 1.00 | 238,161,176,338 | composed |
| 33 | circle | blue | 1.00 | 442,388,108,110 | composed |
| 34 | square | red | 1.00 | 583,388,95,95 | composed |
| 35 | triangle | green | 1.00 | 713,393,105,101 | composed |
| 36 | rectangle | orange | 1.00 | 858,408,122,69 | composed |

Extra: `a boy in front of a girl` -> layout `overlap`, 2 objects

Extra: `two large apples above a small orange` -> layout `column`, 3 objects

### Required assets are never silently replaced

`one red apple and a banana` -> **refused**: [{'request': 'a banana', 'reason': "nothing in the catalog is called 'banana'"}]

`a blue apple` -> **refused**: [{'request': 'blue apple', 'reason': 'no blue apple in the catalog (found: red apple)'}]

Catalog entries after the refusals: 15 (nothing was created for them).
