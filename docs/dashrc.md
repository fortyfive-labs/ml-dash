# `.dashrc` Reference

A `.dashrc` file is a YAML configuration file that controls how experiments and comparisons are displayed in the [Dashboard](dashboard.md). Its behavior depends on where it is placed.

## File Placement

| Location | Type | Controls |
|----------|------|----------|
| Inside an experiment folder | Experiment `.dashrc` | Charts for that single experiment |
| Inside any other folder | Compare `.dashrc` | Cross-experiment comparison charts |

---

## Experiment `.dashrc` vs Compare `.dashrc`

| Feature | Experiment | Compare |
|---------|-----------|---------|
| `metrics` section | ✓ Used | ✗ Ignored |
| `ctype: line` | ✓ Used | ✓ Used |
| `ctype: image` | ✓ Used | ✗ Ignored |
| `ctype: video` | ✓ Used | ✗ Ignored |
| `series` array | ✗ Ignored | ✓ Used |

---

## Complete Field Reference

### Top-Level Fields

```yaml
metrics:       # Experiment .dashrc only
  fields:
    - "*"

charts:
  - ...
```

| Field | Type | Description |
|-------|------|-------------|
| `metrics` | object | Controls which metrics appear in the experiment header. Experiment `.dashrc` only. |
| `metrics.fields` | `string[]` | Glob patterns for metric names. `"*"` matches all. Supports prefixes like `"train*"`. |
| `charts` | `ChartConfig[]` | Ordered list of charts to display. |

---

### `ChartConfig` — Common Fields

All chart types share these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ctype` | `"line" \| "image" \| "video"` | Yes | Chart type |
| `title` | `string` | No | Title displayed above the chart |

---

### `ChartConfig` — Line Chart Fields

```yaml
- ctype: line
  title: My Chart
  xKey: step
  yKey: train.loss         # or yKeys: [...]
  xLabel: Steps
  yLabel: Loss
  bins: 500
  xFormat: null
  yFormat: null
  xTicks: 10
  yTicks: 5
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `xKey` | `string` | `"default"` | X-axis metric name. See [xKey values](#xkey-values) below. |
| `yKey` | `string` | — | Single y-axis metric. Mutually exclusive with `yKeys`. |
| `yKeys` | `string[]` | — | Multiple y-axis metrics on the same chart. Mutually exclusive with `yKey`. |
| `series` | `SeriesConfig[]` | — | Per-series config. **Compare `.dashrc` only.** Mutually exclusive with `yKey` and `yKeys`. Each series issues one API request. |
| `xLabel` | `string` | xKey name | X-axis label |
| `yLabel` | `string` | metric name | Y-axis label |
| `bins` | `number` | `200` | Downsample target point count. Higher = more detail, slower render. |
| `xFormat` | `string \| null` | `null` | Format specifier for x-axis ticks (passed to backend). |
| `yFormat` | `string \| null` | `null` | Format specifier for y-axis ticks (passed to backend). |
| `xTicks` | `number` | auto | Number of x-axis grid lines. |
| `yTicks` | `number` | auto | Number of y-axis grid lines. |

**`yKey` and `yKeys` are mutually exclusive** — use exactly one. (`series` is available in compare `.dashrc` only.)

---

### `SeriesConfig` — Per-Series Fields

Used inside `series` arrays in compare `.dashrc` files only.

| Field | Type | Description |
|-------|------|-------------|
| `prefix` | `string` | Experiment path prefix. The backend resolves this to matching experiments. |
| `experimentIds` | `string[]` | Explicit list of experiment IDs to include in this series. |
| `experimentId` | `string` | Single experiment ID. Shorthand for `experimentIds: [id]`. |
| `label` | `string` | Legend label for this series. |
| `color` | `string` | Line color as a CSS hex string (e.g. `"#5470c6"`). Explicit colors are always preserved. |
| `dash` | `string` | SVG stroke-dasharray string. Omit for a solid line. |
| `xKey` | `string` | Overrides the chart-level `xKey` for this series only. |
| `yKey` | `string` | Overrides the chart-level `yKey` for this series only. |

---

### `ChartConfig` — Image Chart Fields

```yaml
- ctype: image
  title: Frames
  glob: "frames/**/*.png"
  nCols: 4
  nRows: 2
  limit: 8
  sort: date
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `glob` | `string` | `"**/*.{png,jpg,jpeg,gif,webp}"` | Glob pattern to match files within the experiment folder. |
| `nCols` | `number` | `3` | Number of grid columns. |
| `nRows` | `number` | `2` | Number of grid rows. |
| `limit` | `number` | `nCols × nRows` | Maximum number of files to display. |
| `sort` | `"name" \| "date" \| "size"` | `"date"` | File sort order. |

---

### `ChartConfig` — Video Chart Fields

```yaml
- ctype: video
  title: Rollouts
  glob: "videos/**/*.mp4"
  nCols: 2
  nRows: 2
  limit: 4
  sort: name
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `glob` | `string` | `"**/*.{mp4,webm,mov}"` | Glob pattern to match files within the experiment folder. |
| `nCols` | `number` | `3` | Number of grid columns. |
| `nRows` | `number` | `2` | Number of grid rows. |
| `limit` | `number` | `nCols × nRows` | Maximum number of files to display. |
| `sort` | `"name" \| "date" \| "size"` | `"date"` | File sort order. |

---

## `xKey` Values

The `xKey` field specifies which metric is used as the x-axis.

| Value | Behavior |
|-------|----------|
| `"default"` | Backend auto-selects in priority order: `timestamp` → `step` → `epoch`. Falls back to data index if none exist. |
| `"step"` | Uses the `step` field logged alongside each metric. |
| `"epoch"` | Uses the `epoch` field logged alongside each metric. |
| `"timestamp"` | Uses the Unix timestamp recorded automatically by ML-Dash. |
| Any other string | Uses that metric name as the x-axis value. |

---

## Series Colors and Dash Patterns

When `color` is not specified, series are assigned colors automatically from a fixed palette.

### Default Color Palette

| Index | Color | Hex |
|-------|-------|-----|
| 0 | Blue | `#5470c6` |
| 1 | Green | `#91cc75` |
| 2 | Yellow | `#fac858` |
| 3 | Red | `#ee6666` |
| 4 | Sky | `#73c0de` |
| 5 | Teal | `#3ba272` |
| 6 | Orange | `#fc8452` |
| 7 | Purple | `#9a60b4` |
| 8 | Pink | `#ea7ccc` |

For more than 9 series, colors cycle with a **+40° hue rotation** per round, combined with cycling dash patterns.

### Dash Pattern Examples

| Value | Appearance |
|-------|-----------|
| *(omitted)* | Solid line |
| `"4 4"` | Short dashes |
| `"8 4"` | Long dashes |
| `"8 4 2 4"` | Dash-dot |

Explicit `color` values are **always preserved** regardless of series index.

---

## Validation

`.dashrc` files are validated in two stages:

1. **Frontend** — YAML syntax is checked immediately (300ms debounce). Parse errors are shown inline in the editor.
2. **Backend** — Field values are validated when data is fetched. Errors are returned in the API response and displayed below the affected chart.

---

## Quick Reference

```yaml
# Experiment .dashrc — full template
metrics:
  fields:
    - "*"

charts:
  - ctype: line
    title: Chart title
    xKey: default        # or: step, epoch, timestamp, any metric name
    yKey: train.loss     # single metric  ──┐
    # yKeys:             # multiple metrics ┤ pick one
    #   - train.loss     #                  │
    #   - eval.loss      #                 ─┘
    xLabel: Steps
    yLabel: Loss
    bins: 500
    xTicks: 10
    yTicks: 5

  - ctype: image
    glob: "**/*.png"
    nCols: 4
    nRows: 2
    sort: date

  - ctype: video
    glob: "**/*.mp4"
    nCols: 2
    nRows: 2

# Compare .dashrc — series example
charts:
  - ctype: line
    xKey: default
    yKey: train.loss
    series:
      - prefix: alice/project/run-a
        label: Run A
        color: "#5470c6"
      - prefix: alice/project/run-b
        label: Run B
        color: "#ee6666"
        dash: 4 4
      - experimentIds:
          - exp_abc123
          - exp_def456
        label: Seed Group
        color: "#91cc75"
```
