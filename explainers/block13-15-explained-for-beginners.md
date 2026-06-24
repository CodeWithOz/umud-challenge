# Blocks 13, 14, and 15, Explained From the Ground Up

A beginner-friendly walkthrough of what we did after Block 9, why we did it, and
what the leaderboard taught us.

The short version:

- **Block 13** tried a different way to measure the images directly, without using
  the segmentation models for everything. It scored **1.87066**, which was worse.
- **Block 14** kept that direct-measurement idea, but corrected its obvious scale
  problems. It scored **0.96243**, which became the new best at the time.
- **Block 15** blended the corrected direct measurements with the older model-based
  measurements. It scored **0.93837**, the current best after these blocks.

That improved the score from **1.04862** to **0.93837**, but it did **not** reach
the goal of **below 0.6**.

> **Style note:** this document follows the same approach as the Block 9 explainer:
> plain language first, definitions before jargon, small examples where useful, and
> an honesty section that says what we learned even when the score did not hit the
> target.

---

## Table of Contents

1. Where we were after Block 9
2. The problem Blocks 13-15 were trying to solve
3. Block 13: direct image geometry
4. Why Block 13 was not enough
5. Block 14: calibrating the direct measurements
6. Block 15: blending two imperfect predictors
7. What actually happened on the leaderboard
8. What the results taught us
9. What should come next
10. Glossary

---

## Part 1 — Where we were after Block 9

After Block 9, the best score was about **1.04862**.

That was already a big improvement over the older scores around **1.8**, but it was
still not close enough to the target of **below 0.6**.

The important Block 9 lesson was:

> Our old pipeline was mostly getting the *typical middle values* right, but it was
> weak at getting the right value for each individual image.

That matters because the competition gives us one ultrasound image at a time and
asks for three numbers:

| Target | Plain meaning | Unit |
|--------|---------------|------|
| **PA** | Pennation angle: the angle of the muscle fibres | degrees |
| **FL** | Fascicle length: how long a fibre bundle is | millimetres |
| **MT** | Muscle thickness: the gap between the two tissue sheets | millimetres |

Block 9 mostly improved the score by moving predictions toward sensible centres.
For example, instead of predicting a very low pennation angle for every image, it
used a more realistic value around 13-18 degrees.

That helped a lot. But a centred guess can only go so far.

To get from about **1.05** to **below 0.6**, we likely need better **per-image
signal**.

Per-image signal means: image A gets a different prediction from image B for a real
reason, not just noise.

---

## Part 2 — The Problem Blocks 13-15 Were Trying to Solve

The old model-based pipeline was called **segment-then-measure**.

That means:

1. Use a model to draw masks on the image.
2. Measure geometry from those masks.
3. Convert the measurements into PA, FL, and MT.

A **mask** is just a black-and-white image where white means "this pixel belongs to
the structure we care about."

This worked, but it had a major weakness:

> The fascicle part of the system was not giving us strong per-image PA and FL
> information.

In plain terms: the model could produce numbers, but the numbers did not seem to
track each image well enough.

So Blocks 13-15 asked a different question:

> Can we extract useful measurements directly from the ultrasound image itself,
> without relying so heavily on the learned segmentation masks?

This is why we tried a public "quick and dirty" style approach.

"Quick and dirty" does not mean careless here. It means:

- use image brightness and image geometry directly,
- use simple rules instead of training another neural network,
- get a fast test of whether the image itself contains stronger PA/FL/MT signal.

---

## Part 3 — Block 13: Direct Image Geometry

Block 13 was the raw direct-measurement attempt.

Instead of starting from model masks, it looked at the ultrasound image and tried to
estimate the muscle architecture directly.

The rough idea was:

1. Find the useful ultrasound area inside the image.
2. Estimate the two aponeurosis bands, meaning the upper and lower tissue sheets.
3. Estimate the fibre angle from image texture.
4. Use those pieces to calculate:
   - **MT**, the gap between the tissue sheets,
   - **PA**, the fibre angle,
   - **FL**, using the idea that fascicle length depends on thickness and angle.

The relationship behind the last point is:

```
fascicle length is roughly muscle thickness divided by sin(angle)
```

You do not need the trigonometry details. The plain-language version is:

> If the muscle is thick and the fibres are shallow, the fibre path is longer. If the
> fibres are steeper, the fibre path is shorter.

### Why this was worth trying

The old model pipeline had one big weakness: it mostly reused the same fascicle
information every time.

Block 13 gave us a totally different source of information.

That is valuable because even a rough independent method can teach us something:

- If it scores well, the direct image method is useful.
- If it scores badly but differently, we can still use it as a clue.
- If it has better per-image movement but wrong scale, calibration might rescue it.

### What Block 13 predicted

The raw Block 13 output had these typical values:

| Target | Typical raw Block 13 value |
|--------|----------------------------:|
| PA | about **11.31°** |
| FL | about **118.9 mm** |
| MT | about **24.37 mm** |

Those numbers were very different from the Block 9-style predictions:

| Target | Block 9-style area | Block 13 raw area |
|--------|-------------------:|------------------:|
| PA | around **18°** | around **11°** |
| FL | around **74-77 mm** | around **119 mm** |
| MT | around **20-22 mm** | around **24 mm** |

That difference was both good and bad.

Good: it meant Block 13 really was a different measurement path.

Bad: the raw values looked too far away from what past leaderboard work had suggested
was likely.

### Block 13 result

Block 13 scored:

```
1.87066
```

That was much worse than the current best.

So the raw direct-measurement method was not enough.

---

## Part 4 — Why Block 13 Was Not Enough

Block 13 had two problems:

1. The centre was wrong.
2. The spread was too large.

Let's define those.

The **centre** is the typical value. We usually look at the **median**, which means
the middle value after sorting.

The **spread** is how scattered the values are. If predictions range from 60 to 90,
that is tighter than predictions ranging from 30 to 200.

Block 13's fascicle length had a very wide spread:

- minimum around **30 mm**
- maximum around **200 mm**
- typical value around **119 mm**

That is a lot of movement.

Movement is only good if it matches the truth. If the movement is mostly noise, the
score gets worse.

This gave us the Block 14 idea:

> Keep the per-image movement from Block 13, but pull it toward more believable
> centres and reduce how wildly it moves.

That operation is called **calibration**.

Calibration means: adjust the raw predictions so their scale and centre make more
sense.

---

## Part 5 — Block 14: Calibrating the Direct Measurements

Block 14 used the same raw direct-measurement output as Block 13, but applied a fixed
correction.

The correction was:

```text
PA = raw PA + 5 degrees
FL = pull raw FL toward about 76.9 mm
MT = pull raw MT toward about 19.76 mm
```

More exactly:

```text
PA = clip(raw_PA + 5, 5, 45)
FL = 76.9  + 0.20 * (raw_FL - 118.9114)
MT = 19.76 + 0.20 * (raw_MT - 24.3678)
```

Let's translate that.

### The PA correction

Raw Block 13 PA was typically about **11.31°**.

Adding 5 degrees moved it to about **16.31°**.

That put it in the same broad range that Block 9 had shown was much better than the
old low-angle predictions.

The `clip(raw_PA + 5, 5, 45)` part means:

- never go below 5 degrees,
- never go above 45 degrees.

That matches the competition's plausible reference range.

### The FL and MT correction

For FL and MT, we used a simple "pull toward the centre" rule.

Example:

Suppose raw FL is 140 mm.

The raw centre was about 118.9 mm. The target centre was 76.9 mm.

The formula says:

```text
new FL = 76.9 + 0.20 * (140 - 118.9)
       = 76.9 + 0.20 * 21.1
       = 76.9 + 4.22
       = 81.12 mm
```

So the prediction still remembers that this image was above the raw middle, but only
keeps **20%** of that extra movement.

That is called **shrinkage**.

Plain-language definition:

> Shrinkage means keeping a little of the model's per-image variation, while pulling
> most of the prediction back toward a safer centre.

### Why 20%?

Block 13 moved too much.

The earlier Block 9 work suggested that simple centred predictions were already much
safer than noisy wide-spread predictions.

So Block 14 made a conservative choice:

- do not throw away all per-image movement,
- but keep only a small amount of it.

That is why FL and MT used `0.20`.

### What Block 14 predicted

After calibration, the typical values became:

| Target | Typical Block 14 value |
|--------|-----------------------:|
| PA | about **16.31°** |
| FL | about **76.9 mm** |
| MT | about **19.76 mm** |

Those are much closer to the values that previous leaderboard feedback had suggested
were promising.

### Block 14 result

Block 14 scored:

```
0.96243
```

That was a real improvement:

| Previous best | Block 14 |
|--------------:|---------:|
| **1.04862** | **0.96243** |

So Block 14 worked.

But it did not work enough. The goal is still below **0.6**.

---

## Part 6 — Block 15: Blending Two Imperfect Predictors

Block 15 tried a blend.

A **blend** means averaging two different predictions together.

For example, if method A predicts FL = 80 and method B predicts FL = 70, then a 50/50
blend predicts:

```text
(80 + 70) / 2 = 75
```

Block 15 did not use a 50/50 blend. It used:

```text
70% Block 14 calibrated quick-dirty
30% cxs8 model pipeline
```

In shorthand:

```text
Block 15 = 0.70 * Block 14 + 0.30 * cxs8
```

### What is cxs8?

`cxs8` means the convnext-small aponeurosis model trained for 8 epochs.

You do not need to know what "convnext-small" means internally. In this project it
is one of the model families we tried for predicting the aponeurosis mask.

The important part:

- Block 14 came from direct image geometry.
- `cxs8` came from the older segment-then-measure model pipeline.

So Block 15 combined two different kinds of mistakes.

### Why blending can help

Blending helps when two methods are wrong in different ways.

Tiny example:

| Image | Truth | Method A | Method B | 50/50 Blend |
|-------|------:|---------:|---------:|------------:|
| 1 | 20 | 18 | 24 | 21 |
| 2 | 20 | 23 | 17 | 20 |

Method A and Method B are both imperfect, but their errors partly cancel out.

That is the hope with a blend.

### Why this particular blend?

Block 14 was better than the model-only pipeline, so it got the larger weight:

- **70%** direct calibrated geometry,
- **30%** model-based geometry.

The idea was:

> Keep most of the new Block 14 signal, but add back some stability from the older
> model pipeline.

### What Block 15 predicted

Block 15's typical values were:

| Target | Typical Block 15 value |
|--------|-----------------------:|
| PA | about **16.82°** |
| FL | about **75.15 mm** |
| MT | about **20.41 mm** |

These are very close to Block 14, but slightly nudged toward the model pipeline.

### Block 15 result

Block 15 scored:

```
0.93837
```

That beat Block 14:

| Block | Score |
|-------|------:|
| Block 14 | **0.96243** |
| Block 15 | **0.93837** |

So the blend helped.

But again, it did not help enough to reach **0.6**.

---

## Part 7 — What Actually Happened on the Leaderboard

Here are the three blocks together:

| Block | What changed | Score | What it means |
|-------|--------------|------:|---------------|
| **13** | Raw direct image geometry | **1.87066** | Different method, but raw scale/centre were bad |
| **14** | Calibrated direct geometry | **0.96243** | Big improvement; calibration rescued part of the idea |
| **15** | 70% calibrated direct geometry + 30% model pipeline | **0.93837** | Small extra gain; new best |

The score moved like this:

```text
1.04862  ->  0.96243  ->  0.93837
old best     Block 14     Block 15
```

That is progress, but not a breakthrough.

### A note about the scheduled submissions

These blocks were supposed to submit automatically after the daily quota reset.

The scheduled jobs did start, but they failed before submitting because Kaggle's
access-token command returned an HTTP 429 error. HTTP 429 means "too many requests"
or "rate limited."

So the notebooks had run, but the scored submissions did not appear until we submitted
them manually on June 24.

---

## Part 8 — What the Results Taught Us

### Lesson 1: Raw direct geometry is not trustworthy by itself

Block 13 was too far off.

That tells us the image-only measurement rules are not accurate enough in raw form.

The method may still contain useful information, but it needs strong correction.

### Lesson 2: Calibration still matters a lot

Block 14 turned a bad raw score (**1.87066**) into a strong score (**0.96243**).

That is a huge change without retraining a model.

Plain-language takeaway:

> The direct method was not useless. It was badly scaled.

### Lesson 3: Per-image signal exists, but it is weaker than hoped

If Block 14 had strong per-image signal, it might have scored much closer to 0.6.

It did improve the score, which suggests it added something real.

But it did not improve enough, which suggests a lot of its per-image movement is still
noise.

### Lesson 4: The blend helped, but only a little

Block 15 improved from **0.96243** to **0.93837**.

That means the model-based predictions added a small amount of useful correction.

But the gain was only about **0.024**.

That is not the size of improvement needed to reach **0.6**.

### Lesson 5: The old tracking proxy was too optimistic here

Before scoring, Block 15 looked much better under the local tracking proxy.

It was expected to be around **0.485** by that rough proxy, but the real score was
**0.93837**.

That does not mean the proxy was worthless. It correctly pointed toward Block 14/15
being better than the old best.

But it was too optimistic about how much better.

Plain-language takeaway:

> The proxy was useful for direction, not for exact score prediction.

---

## Part 9 — What Should Come Next

The next work should not simply repeat the same blend with tiny weight changes.

Why?

Because Block 14 and Block 15 are close:

```text
Block 14 = 0.96243
Block 15 = 0.93837
```

Changing the blend weight might shave off a little more, but it is unlikely to jump
from **0.938** to **below 0.6** by itself.

The more useful next step is to diagnose which target still dominates the error:

- Is PA still too high or too low?
- Is FL still centred wrong?
- Is MT spread still hurting us?
- Are the direct measurements helping only certain image sizes?

The most promising direction is probably:

1. use the new Block 13-15 scores as real feedback,
2. refit the calibration assumptions,
3. build one or two stronger candidates that change the actual prediction behaviour,
   not just the blend weight by a tiny amount.

In plain terms:

> Blocks 13-15 proved that direct image geometry can help, but also proved that the
> current version is not accurate enough. The path to below 0.6 likely needs either a
> better direct measurement method, a better way to choose per-image corrections, or a
> stronger model artifact to blend with.

---

## Part 10 — Glossary

**Aponeurosis / apo**  
A sheet of connective tissue around the muscle. Muscle thickness is the gap between
the upper and lower aponeurosis.

**Blend**  
Combining two predictions by weighted averaging. Example: 70% of one method plus 30%
of another.

**Calibration**  
Adjusting predictions so their centre and scale better match what we believe the
truth looks like.

**Centre**  
The typical value of a set of predictions. We usually use the median because it is
robust to extreme values.

**cxs8**  
The convnext-small aponeurosis model trained for 8 epochs. It is part of the older
segment-then-measure pipeline.

**Direct image geometry**  
A method that measures patterns from the ultrasound image itself instead of relying
entirely on model-predicted masks.

**FL**  
Fascicle length, measured in millimetres.

**Mask**  
A black-and-white image produced by a segmentation model. White pixels mark the
structure the model thinks it found.

**Median**  
The middle value after sorting. If the values are `[10, 12, 14, 16, 100]`, the
median is `14`.

**MT**  
Muscle thickness, measured in millimetres.

**PA**  
Pennation angle, measured in degrees.

**Per-image signal**  
Useful information that makes the prediction different for each image in a way that
matches the truth.

**Proxy**  
A local estimate of leaderboard score. Useful for choosing experiments, but not a
replacement for the real leaderboard.

**Quick-dirty / quickdirty**  
The direct-measurement style used in Block 13. The name means fast and simple, not
careless.

**Segment-then-measure**  
First draw masks with a model, then measure geometry from those masks.

**Shrinkage**  
Pulling predictions toward a safer centre while keeping a smaller amount of their
per-image movement.

**Spread**  
How scattered the predictions are. A small spread means predictions are tightly
clustered; a large spread means they vary widely.

