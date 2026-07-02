# Road‑Research Graph Module – Detailed Documentation

> **Purpose**: Convert raster road predictions (binary masks and optional probability maps) into a clean, vector‑based road graph.  The module is fully modular; each stage can be toggled via the command‑line interface.

---

## Table of Contents
1. [High‑level Flow](#high-level-flow)
2. [Entry Point – `run_pipeline.py`](#entry-point-run_pipelinepy)
3. [Core Sub‑modules](#core-sub-modules)
   - [Endpoint Detection (`endpoint_detection.py`)](#endpoint-detection)
   - [Branch Detection (`branch_detection.py`)](#branch-detection)
   - [Candidate Generation (`candidate_pairs.py`)](#candidate-generation)
   - [Cost Function (`scoring.py`)](#cost-function)
   - [Search – Bidirectional A* (`astar.py`)](#bidirectional-astar)
   - [Topology Repair (`topology_repair.py`)](#topology-repair)
   - [Validation (`validation.py`)](#validation)
   - [Debug Visualisation (`debug_viz.py`)](#debug-visualisation)
   - [Graph Construction & Simplification (`graph_builder.py`, `graph_healing.py`)](#graph-construction)
4. [Random‑sample driver – `run_random_samples.py`](#random-samples)
5. [Data Flow Diagram (Mermaid)](#mermaid-diagram)
6. [Running the Pipeline](#running-the-pipeline)
7. [Extending / Customising](#extending)
---

## <a name="high-level-flow"></a>1. High‑level Flow
```mermaid
flowchart TD
    A[Mask + (optional) Prob Map] --> B[Pre‑process & Skeletonize]
    B --> C{Repair?}
    C -- Yes --> D[TopologyRepair]
    D --> E[Repaired Skeleton]
    C -- No --> E[Original Skeleton]
    E --> F[GraphBuilder]
    F --> G[Graph (nodes/edges)]
    G --> H{Simplify?}
    H -- Yes --> I[GraphHealing (prune, collapse, …)]
    H -- No --> I[Graph (unchanged)]
    I --> J[Output: JSON, visualisations]
```
*The decision points (`Repair?`, `Simplify?`) correspond to CLI flags (`--no‑repair`, `--no‑simplify`).*

---

## <a name="entry-point-run_pipelinepy"></a>2. Entry Point – `run_pipeline.py`
File: **[run_pipeline.py](file:///d:/S%20files/Road%20research/graph_module/run_pipeline.py)**

| Step | Description | Key Code Snippets |
|------|-------------|-------------------|
| **Argument parsing** | Handles mask path, optional probability map, output directory, and a series of flags that enable/disable later stages. | `parser.add_argument('--mask', required=True)` … |
| **Load data** | Binary mask (`cv2.imread(...,0)`) → `mask`.  Optional probability map (`np.load`). | `mask = cv2.imread(mask_path,0)` |
| **Skeletonisation** | `skeleton = skeletonize(mask > 0)` (using `skimage.morphology`). | `skeleton = skeletonize(mask > 0)` |
| **Repair (optional)** | Instantiates `TopologyRepair` with the probability map and runs it. | `if args.prob_map: repaired = TopologyRepair(...).run()` |
| **Graph building** | Calls `graph_builder.build_graph(skel, ...)` to obtain a NetworkX‑compatible graph. | `graph = build_graph(skeleton, ...)` |
| **Simplification (optional)** | Calls `graph_healing.simplify_graph` with the various `--no‑*` flags. | `if not args.no_simplify: simplify_graph(graph, args)` |
| **Output** | Writes `graph.json`, optional PNG visualisations, and a human‑readable summary. | `save_graph(graph, out_dir)` |

> **Modularity note**: Every stage receives only the data it needs, making it possible to run them independently.

---

## <a name="core-sub-modules"></a>3. Core Sub‑modules
Below each sub‑module is described with its public API, the role it plays in the pipeline, and highlights of the recent algorithmic improvements.

### <a name="endpoint-detection"></a>a. Endpoint detection – `endpoint_detection.py`
File: **[endpoint_detection.py](file:///d:/S%20files/Road%20research/graph_module/endpoint_detection.py)**
* **`get_endpoints(skel)`** – returns a list of `(y,x)` pixels that have exactly one 8‑connected neighbour. These are the *dead‑ends* of the road network.
* **`get_connected_components(skel)`** – returns a labeled image (`labeled_comp, num_comp`) where each skeleton component receives a unique integer label. Used to avoid connecting points already belonging to the same component.
* **`estimate_endpoint_tangent(ep, skel, branch_mask)`** – extracts a small patch around `ep`, runs PCA on the foreground pixels, and returns a unit vector `t` that approximates the local road direction.  This tangent is later used for:
  * Scoring candidates (alignment of tangents) and
  * The **endpoint‑tangent consistency** term in the cost function.

---

### <a name="branch-detection"></a>b. Branch detection – `branch_detection.py`
File: **[branch_detection.py](file:///d:/S%20files/Road%20research/graph_module/branch_detection.py)**
* **`get_branch_points(skel)`** – returns all skeleton pixels with more than two neighbours (junctions).
* **`extract_branches(skel, branch_mask)`** – flood‑fills the skeleton between branch points / endpoints, labeling each continuous segment with a unique integer. The resulting `labeled_branches` map is used for **endpoint‑to‑branch** candidate generation.

---

### <a name="candidate-generation"></a>c. Candidate Generation – `candidate_pairs.py`
File: **[candidate_pairs.py](file:///d:/S%20files/Road%20research/graph_module/candidate_pairs.py)**
**What changed**: the previous straight‑line average‑probability heuristic has been **removed**.  Candidates are now ranked purely on geometric and topological cues.

#### Functions
* **`generate_candidate_pairs(endpoints, tangents, labeled_components, search_radius, min_score)`** – builds *endpoint‑to‑endpoint* candidates. For each unordered pair `(i,j)`:
  * Skip if both endpoints belong to the same component (`labeled_components`).
  * Compute a **confidence score** via `score_candidate` (see below).
  * Keep the candidate if `score >= min_score`.
* **`generate_endpoint_to_branch_candidates(endpoints, tangents, labeled_components, labeled_branches, search_radius, min_score)`** – builds *endpoint‑to‑branch* candidates. For each endpoint we find the **closest branch pixel** (within `search_radius`). The score is again derived from `score_candidate`.

#### Scoring (`score_candidate`)
```python
score = distance_score + 0.5 * max(0, src_tangent·line_dir)
if dst_tangent is not None:
    score += 0.5 * max(0, dst_tangent·(-line_dir))
    score += 0.5 * max(0, src_tangent·(-dst_tangent))
else:
    score += 1.0            # boost for branch targets
```
* `distance_score` = `1 – (dist / search_radius)` (higher when the endpoints are closer).
* Tangent‑alignment terms reward vectors that point **towards** the partner and that are roughly opposite to each other (good for a smooth connection).
* The final list of candidates is **sorted descending by `score`** before feeding the search algorithm.

---

### <a name="cost-function"></a>d. Cost Function – `scoring.py`
File: **[scoring.py](file:///d:/S%20files/Road%20research/graph_module/scoring.py)**
`CostFunction` now combines six weighted terms:
1. **Euclidean distance** (`w_dist`).
2. **Inverse probability** (`w_prob * (1‑p)`).  Low‑confidence pixels incur an additional penalty (`w_low_conf`).
3. **Direction change** (`w_dir`).  Measured as `1 – cos(previous_dir, new_dir)`.
4. **Curvature** (`w_curve`).  Penalises sharp turns (`cos < 0`).
5. **Target alignment** (`w_target_align`).  Rewards moving towards the final target point.
6. **Endpoint‑tangent consistency** (`w_ep_tangent`).
   * Near the source (`src`) we compare the step direction with the source tangent.
   * Near the destination (`target_pt`) we compare with the *negative* of the destination tangent (entering the target).
   * The influence fades linearly with distance (within 20 px by default).

All weights are exposed via `TopologyRepair.__init__` – see the constructor in `topology_repair.py`.

---

### <a name="bidirectional-astar"></a>e. Search – Bidirectional A* (`astar.py`)
File: **[astar.py](file:///d:/S%20files/Road%20research/graph_module/astar.py)**
#### Public API
```python
run_bidirectional_astar(src, target, target_pt, cost_func,
                         img_shape, margin, traversable_mask=None)
```
* **`src`** – start pixel `(y,x)`.
* **`target`** – either a single pixel `(y,x)` **or** a list of branch pixels.
* **`target_pt`** – the *focus* of the ellipse (the exact endpoint we are trying to reach).
* **`margin`** – the extra allowance added to the ellipse radius; grows with the stage (5→10→15 px).
* **`traversable_mask`** – Boolean mask limiting expansion to:
  * `prob_map > prob_threshold` **or**
  * `distance to current skeleton ≤ margin`.

#### Adaptive Elliptical Region
A node `v` is expanded **only if**
```
dist(src, v) + dist(v, target_pt) < dist(src, target_pt) + margin
```
This automatically widens for larger gaps while keeping the search focused.

#### Heuristic
Standard Euclidean distance to the nearest goal (or to `target_pt` when the goal is a branch).  Guarantees admissibility.

#### Returned values
* `path` – list of `(y,x)` coordinates from `src` to the first reached goal.
* `explored_count` – number of nodes popped from the priority queue (useful for debugging).
* `final_cost` – accumulated cost of the chosen path (passed to `DebugLogger`).

---

### <a name="topology-repair"></a>f. Topology Repair – `topology_repair.py`
File: **[topology_repair.py](file:///d:/S%20files/Road%20research/graph_module/topology_repair.py)**
**High‑level algorithm** (three progressive stages):
1. **Detect endpoints** and **branch points** on the *current* skeleton.
2. **Estimate tangents** for each endpoint.
3. **Generate candidates** (ep‑ep and ep‑branch) and sort by the **confidence score** from `candidate_pairs`.
4. **Build a traversability mask** (`prob > threshold OR within margin of skeleton`).
5. For each candidate (highest confidence first):
   * Construct a `CostFunction` that knows the source/target tangents.
   * Run `run_bidirectional_astar` with the appropriate `margin`.
   * Validate the returned path via `validation.validate_path`.
   * If valid, **merge** the path into the skeleton, re‑skeletonise, and refresh connectivity labels.
6. Continue until all candidates are exhausted for the stage, then move to the next (larger) stage.

#### Logging & Debugging
`DebugLogger.log_candidate` now records:
* `score`
* `explored_nodes`
* `final_cost`
* `reason` (validation outcome)
The log is saved as `graph_debug/repair_log.json` and visualised in `repair_debug_viz.png`.

---

### <a name="validation"></a>g. Validation – `validation.py`
File: **[validation.py](file:///d:/S%20files/Road%20research/graph_module/validation.py)**
A path is **accepted** only when the following **multi‑metric** checklist passes:
| Metric | Computation | Threshold (default) |
|--------|--------------|--------------------|
| **Average probability** | `np.mean(prob_map[y,x] for (y,x) in path)` | `≥ min_avg_prob` (0.3) |
| **Median probability** | `np.median(...)` | `≥ 0.8 × min_avg_prob` |
| **Low‑confidence ratio** | fraction of pixels with `p < prob_threshold` | `≤ max_low_conf_ratio` (0.5) |
| **Path efficiency** | `len(path) / euclidean_dist(src, dst)` | `≤ 2.0` |
| **Curvature** | count of >120° turns (`cos < -0.5`) | `≤ 1` |
| **Critical low avg** | `avg_prob < min_avg_prob/2` | immediate reject |

*The path is rejected only if **≥ 2** non‑critical failures occur, or a critical failure triggers immediate rejection.*

---

### <a name="debug-visualisation"></a>h. Debug Visualisation – `debug_viz.py`
File: **[debug_viz.py](file:///d:/S%20files/Road%20research/graph_module/debug_viz.py)**
The logger stores a compact JSON **summary** (stage, src/dst, distance, score, path length, explored nodes, final cost, validity, reason).  The visualiser overlays:
* **Accepted paths** – green solid lines.
* **Rejected paths** – red dashed lines with the failure reason printed near the middle.
* Endpoint markers (green/red circles).
* The underlying probability map for context (grayscale with 0.5 α).
The image is saved as `graph_debug/repair_debug_viz.png`.

---

### <a name="graph-construction"></a>i. Graph Construction & Simplification
* **`graph_builder.py`** – walks the (repaired) skeleton, creates a node for every endpoint/branch, and an edge for each continuous segment.  The result is a **NetworkX‑compatible** graph stored as JSON.
* **`graph_healing.py`** – optional post‑processing that:
  * Removes degree‑2 nodes (straight‑line compression).
  * Collapses edges shorter than `--short-edge-threshold`.
  * Deletes tiny cycles (`--tiny-cycle-perimeter`, `--tiny-cycle-radius`).
  * Deduplicates overlapping paths.
  * All of these steps can be disabled via CLI flags (`--no‑simplify`, `--no‑contract-degree2`, etc.).

---

## <a name="random-samples"></a>4. Random‑sample driver – `run_random_samples.py`
File: **[run_random_samples.py](file:///d:/S%20files/Road%20research/run_random_samples.py)**
* Picks *N* random prediction masks from `pred_masks/`.
* Constructs the appropriate command line for **each** image (respecting flags such as `--no‑repair`).
* Executes `run_pipeline.py` via `subprocess.run` and prints a concise summary (nodes, edges, diagnostics).  Useful for bulk testing or benchmarking.

---

## <a name="mermaid-diagram"></a>5. Data‑flow diagram (Mermaid)
```mermaid
flowchart TD
    A[Mask + (optional) Prob Map] --> B[Pre‑process & Skeletonize]
    B --> C{Repair?}
    C -- Yes --> D[TopologyRepair]
    D --> E[Repaired Skeleton]
    C -- No --> E[Original Skeleton]
    E --> F[GraphBuilder]
    F --> G[Graph (nodes/edges)]
    G --> H{Simplify?}
    H -- Yes --> I[GraphHealing]
    H -- No --> I[Graph (unchanged)]
    I --> J[Output: JSON, PNG, logs]
```
---

## <a name="running-the-pipeline"></a>6. Running the Pipeline
```bash
# Activate the virtual environment (if not already active)
source venv/Scripts/activate  # PowerShell: .\venv\Scripts\Activate.ps1

# Basic run on a single mask (with probability map)
python graph_module/run_pipeline.py \
    --mask pred_masks/012345_pred.png \
    --prob-map pred_masks/012345_prob.npy \
    --output-dir graph_output

# Run on a whole directory (random sampling of 20 images)
python run_random_samples.py --samples 20 --seed 42
```
Key flags (most useful):
* `--no-repair` – skip topology repair (useful for speed).
* `--no-simplify` – keep the raw graph.
* `--short-edge-threshold`, `--tiny-cycle-perimeter`, etc. – tune simplification aggressiveness.
* `--compare-no-simplify` – automatically run a second pass without simplification for side‑by‑side comparison.

---

## <a name="extending"></a>7. Extending / Customising
| What you might want to change | Where to edit | Quick tip |
|------------------------------|----------------|-----------|
| **Different weighting** for distance vs. probability | `TopologyRepair.__init__` (or pass weights via CLI) | Adjust `w_dist`, `w_prob`, `w_ep_tangent`, etc. |
| **Alternative heuristics** (e.g., learned edge costs) | `scoring.CostFunction.get_cost` | Add your own term before `total_cost` is returned. |
| **New candidate filters** (e.g., enforce angle limits) | `candidate_pairs.score_candidate` | Insert additional alignment checks. |
| **Different traversal mask** (e.g., incorporate a road‑network prior) | `topology_repair._run_stage` – modify the `traversable_mask` construction. |
| **Export to another graph format** (GeoJSON, Shapefile) | `graph_builder` – after the NetworkX graph is built, use `networkx.write_gpickle` or a custom writer. |

---

## 8. Where to Find the Source Files
| Module | Path (click to open) |
|--------|----------------------|
| Entry point | [run_pipeline.py](file:///d:/S%20files/Road%20research/graph_module/run_pipeline.py) |
| Endpoint detection | [endpoint_detection.py](file:///d:/S%20files/Road%20research/graph_module/endpoint_detection.py) |
| Branch detection | [branch_detection.py](file:///d:/S%20files/Road%20research/graph_module/branch_detection.py) |
| Candidate generation | [candidate_pairs.py](file:///d:/S%20files/Road%20research/graph_module/candidate_pairs.py) |
| Scoring / cost | [scoring.py](file:///d:/S%20files/Road%20research/graph_module/scoring.py) |
| A* search | [astar.py](file:///d:/S%20files/Road%20research/graph_module/astar.py) |
| Topology repair | [topology_repair.py](file:///d:/S%20files/Road%20research/graph_module/topology_repair.py) |
| Validation | [validation.py](file:///d:/S%20files/Road%20research/graph_module/validation.py) |
| Debug logger | [debug_viz.py](file:///d:/S%20files/Road%20research/graph_module/debug_viz.py) |
| Graph building | [graph_builder.py](file:///d:/S%20files/Road%20research/graph_module/graph_builder.py) |
| Graph simplification | [graph_healing.py](file:///d:/S%20files/Road%20research/graph_module/graph_healing.py) |
| Random‑sample driver | [run_random_samples.py](file:///d:/S%20files/Road%20research/run_random_samples.py) |

---

## 9. Final notes
*The pipeline is deliberately **modular**; you can replace any component (e.g., swap the A* for a learned planner) as long as the public function signatures stay the same.*

Feel free to explore the source files via the links above, adjust the weighting parameters, or run the random‑sample driver to see the whole system in action.

---


