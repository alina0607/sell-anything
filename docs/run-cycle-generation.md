# Run-cycle generation provenance

The accepted `art/stage/girl_run_cycle_v3.png` was assembled from two built-in ImageGen outputs, then normalized
to one head width, head height, value and saturation. The animation audit checks the built sprite sheet rather
than trusting these prompts.

## Four-panel edit prompt

```text
Use case: precise-object-edit
Edit target: the immediately preceding four-panel run-cycle sheet.
Primary request: Keep panels 1 and 2 visually unchanged. Correct ONLY panels 3 and 4 so the four panels form a genuinely alternating gait.
Panel 3 must be the exact opposing extreme of panel 1: the arm that is forward in panel 1 must be clearly behind the torso in panel 3, while the other arm must extend forward toward screen-right. The boot/leg that is straight forward toward screen-right in panel 1 must be clearly trailing toward screen-left in panel 3, while the other boot/leg must extend forward toward screen-right. A viewer must be able to identify two opposite arm silhouettes and two opposite leg silhouettes without relying on shading.
Panel 4 must be the opposite crossing midpoint from panel 2: swap which arm and which leg visually overlaps in front.
Lock invariants across all four panels: pixel-identical head dimensions, face, hair silhouette, braid length, torso size, skirt size, total scale, leg length, boot size, watercolor rendering, brightness, saturation and palette. Do not repaint or darken panels 1 and 2. No extra limbs, merged limbs, repeated stride, clasped hands at both extremes, head-size change, background, shadow, labels, grid, or text. Preserve true transparency and four equal horizontal cells.
```

## Opposing-stride edit prompt

```text
Use case: precise-object-edit
Edit target: the immediately preceding four-panel transparent run-cycle sheet.
Change ONLY limb identity/layering in PANEL 3; do not change the silhouette, head, hair, face, torso, skirt, scale, palette, or any other panel.
Critical leg edit in panel 3: the leg extending forward toward SCREEN-RIGHT must become the FAR leg — slightly darker skin, darker sock, partially occluded at the hip. The leg trailing toward SCREEN-LEFT must become the NEAR leg — brighter skin, clearly visible white sock, and visibly layered on top at the hip. This must be the opposite near/far leg ordering from panel 1.
Critical arm edit in panel 3: exchange which anatomical arm is on top. The arm reaching toward SCREEN-RIGHT must originate from the far shoulder and pass behind the chest. The arm swinging toward SCREEN-LEFT must originate from the near shoulder and visibly overlap on top of the torso. This must be the opposite arm overlap ordering from panel 1.
Panel 1 remains stride A; panel 2 remains crossing A; corrected panel 3 is stride B with opposite limb identity; panel 4 remains crossing B.
Preserve exact identical brightness and saturation across the whole character. No pose redesign, no new silhouette, no extra limbs, no missing limbs, no darkening the whole frame, no head-size change, no background, no grid, no text. Keep true transparency.
```
