# Scoop check: zero-shot 3D-aware fruit perception scored on mass

Date: 2026-11-05. Searches were run once each with a general web search tool, so most evidence is
a search snippet, not a read paper. The tag after each source says how much of it was actually read.

Evidence tags: **abstract** = abstract page fetched and read; **snippet** = search-result summary only;
**README** = repository page read; **full PDF** = full text read. No source below was read as a full PDF.

## The paper named in the plan

| Source | Tag | What it is | Relevance |
|---|---|---|---|
| arXiv 2504.07252, Singh, Puhl, Dhakal, Sornapudi, "Few-Shot Adaptation of Grounding DINO for Agricultural Domain" | abstract | Few-shot fine-tuning of Grounding DINO (text encoder replaced by a learned embedding) on plant-weed, counting, insect, fruit counting and remote-sensing sets. Metric is mAP. | 2D detection only. No depth, volume or mass, per the abstract. It is the "prior work is 2D" reference in the research note, and that description is correct. It is few-shot, not zero-shot. |

## Searches for zero-shot fruit volume or mass from RGB-D

| Query | Source | Tag | Finding |
|---|---|---|---|
| zero-shot fruit volume mass estimation RGB-D Grounding DINO SAM | Barakat et al. 2023, durian sizing with zero-shot models (IJTech 6640; MMU repository) | snippet | Grounding DINO then SAM to mask durians for size. Image-based sizing, no RGB-D, no mass. |
| same | "Foundation Model-Based Apple Ripeness and Size Estimation for Selective Harvesting" (2025 preprint listing) | snippet | Grounding DINO plus depth for apple size, six size algorithms. Size, not volume or mass. Closest in using depth with a zero-shot detector. |
| same | Banana hand mass from RGB-D with SAM, Transactions of the CSAE 2025 (aeeisp.com) | snippet | SAM finger masks, 3D features, random forest, R2 0.984, MAPE 5.37%. Supervised regression, not zero-shot detection, and the detector is not compared across models. |
| same | Camellia oleifera fruit, Engineering 2026 | snippet | Grounding DINO with the prompt "fruit", then 3D surface optimisation. Localisation focus. |
| same | DEV Community post, Grounding DINO + SAM for food portions | snippet | A blog tutorial, not a validation. |
| oil palm FFB weight depth camera foundation model | UAV YOLOv8 + random forest for FFB mass (2025); cross-line laser volume (2022); multi-view pixel-count regression | snippet | Oil palm mass work is supervised or classical. None uses a zero-shot foundation model. |
| same | Grape bunch RGB-D: RGB mask projected to depth for volume and weight | snippet | The same "RGB mask to depth to volume" pattern, with a trained segmenter. |
| zero-shot SAM2 Grounding DINO YOLO-World agriculture | Wild blueberry Sensors 2025 and Agronomy 2024 (PMC12694221) | snippet | Compares Grounding DINO and YOLO-World with SAM2 variants. Scored by detection and segmentation metrics, not volume or mass. The closest model-set match. |
| same | arXiv 2411.16196, SAM2 + OpenCLIP fruit detection (SDM-D, MegaFruits) | snippet | Compared with Grounding SAM and YOLO-World on fruit detection. 2D metrics. |

## Verdict

**Not scooped, with caveats.** In what was found, nobody scores Grounding DINO, YOLO-World and SAM2 masks
by the error of a downstream volume-to-mass estimate from RGB-D. Related work scores them by mAP or IoU,
or uses one of them for fruit size. The apple size paper and the banana mass paper are the nearest
neighbours and should be cited and read in full before any write-up.

Caveats:
- Everything is snippet or abstract level. A full-text search of Google Scholar and arXiv listings was not possible here.
- The claim "open for volume and mass" is a negative result from a handful of queries. Treat it as "not found", not "does not exist".
- This repo's own report already records one negative Grounding DINO to SAM2 result (1.99 kg against 1.27 kg, in-sample). The novelty here is the pre-registered, held-out, paired comparison with intervals, not the idea.
- With 10 scored bunches the study can only separate large differences, which limits its publishable claim to a small-n case study.
