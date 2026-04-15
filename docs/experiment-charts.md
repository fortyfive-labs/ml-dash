# Experiment Chart Configuration

Customize the charts displayed in an experiment's Dashboard tab by creating a `.dashrc` file inside the experiment folder.

## Creating and Editing `.dashrc`

If no `.dashrc` exists yet, click **Save as .dashrc** in the Dashboard tab toolbar. This generates a default configuration based on the current auto-generated charts, which you can then edit.

To edit an existing `.dashrc`:

1. Click the **Edit** button in the top-right of the Dashboard tab or click the .dashrc file.
2. A split view opens: YAML editor on the left, live chart preview on the right.
3. Edit — charts update automatically.

```{figure} _static/images/dashrc-editor.png
:alt: .dashrc split editor view
:width: 100%

The `.dashrc` editor: YAML editor on the left with live chart preview on the right. Charts re-render automatically as you edit.
```

## `metrics` — Header Metrics Filter

Controls which metrics appear in the experiment header, displayed as a row of key-value pairs above the charts.

```yaml
metrics:
  fields:
    - "*"          # Show all metrics (default)
    # - "train*"   # Only metrics starting with "train"
    # - "eval*"    # Only metrics starting with "eval"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fields` | `string[]` | `["*"]` | Glob patterns to filter which metrics are shown in the header |

---

## Line Charts (`ctype: line`)

Line charts display time-series metrics tracked with `exp.metrics()`.

### Minimal Example

```yaml
charts:
  - ctype: line
    xKey: step
    yKey: train.loss
```

### Full Example

```yaml
charts:
  - ctype: line
    title: Training & Eval Loss
    xKey: step
    yKeys:
      - train.loss
      - eval.loss
    xLabel: Steps
    yLabel: Loss
    bins: 500
    xTicks: 10
    yTicks: 5
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ctype` | `"line"` | — | Chart type (required) |
| `title` | `string` | metric name | Chart title shown above the chart |
| `xKey` | `string` | `"default"` | Metric name for the x-axis. Use `"default"` to let the backend auto-select (timestamp → step → epoch) |
| `yKey` | `string` | — | Single y-axis metric. Mutually exclusive with `yKeys` |
| `yKeys` | `string[]` | — | Multiple y-axis metrics on the same chart. Mutually exclusive with `yKey` |
| `xLabel` | `string` | xKey value | X-axis label |
| `yLabel` | `string` | metric name | Y-axis label |
| `bins` | `number` | `200` | Number of points to downsample to. Higher values show more detail but are slower to render |
| `xFormat` | `string \| null` | `null` | Format specifier for x-axis tick labels (passed to backend) |
| `yFormat` | `string \| null` | `null` | Format specifier for y-axis tick labels (passed to backend) |
| `xTicks` | `number` | auto | Number of grid ticks on the x-axis |
| `yTicks` | `number` | auto | Number of grid ticks on the y-axis |

### `yKey` vs `yKeys`

Use `yKey` for a single metric:

```yaml
- ctype: line
  xKey: step
  yKey: train.loss
```

Use `yKeys` to plot multiple metrics on the same chart:

```yaml
- ctype: line
  xKey: step
  yKeys:
    - train.loss
    - eval.loss
```

### `xKey: "default"`

When `xKey` is set to `"default"`, the backend automatically selects the best x-axis metric in this priority order:

1. `timestamp`
2. `step`
3. `epoch`

If none of these exist, the data index is used.

---

## Image Charts (`ctype: image`)

Image charts display image files uploaded with `exp.files().save_image()` or any image artifact.

### Example

```yaml
charts:
  - ctype: image
    title: Training Samples
    glob: "frames/**/*.png"
    nCols: 4
    nRows: 2
    sort: date
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ctype` | `"image"` | — | Chart type (required) |
| `title` | `string` | — | Title shown above the grid |
| `glob` | `string` | `"**/*.{png,jpg,jpeg,gif,webp}"` | Glob pattern to match image files within the experiment folder |
| `nCols` | `number` | `3` | Number of columns in the image grid |
| `nRows` | `number` | `2` | Number of rows in the image grid |
| `limit` | `number` | `nCols × nRows` | Maximum number of images to display |
| `sort` | `"name" \| "date" \| "size"` | `"date"` | Sort order for matched files |

---

## Video Charts (`ctype: video`)

Video charts display video files uploaded with `exp.files().save_video()` or any video artifact.

### Example

```yaml
charts:
  - ctype: video
    title: Episode Rollouts
    glob: "videos/**/*.mp4"
    nCols: 2
    nRows: 2
    sort: name
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ctype` | `"video"` | — | Chart type (required) |
| `title` | `string` | — | Title shown above the grid |
| `glob` | `string` | `"**/*.{mp4,webm,mov}"` | Glob pattern to match video files within the experiment folder |
| `nCols` | `number` | `3` | Number of columns in the video grid |
| `nRows` | `number` | `2` | Number of rows in the video grid |
| `limit` | `number` | `nCols × nRows` | Maximum number of videos to display |
| `sort` | `"name" \| "date" \| "size"` | `"date"` | Sort order for matched files |

---

## Complete Example

```yaml
metrics:
  fields:
    - "*"

charts:
  # Training and evaluation loss on one chart
  - ctype: line
    title: Loss
    xKey: step
    yKeys:
      - train.loss
      - eval.loss
    xLabel: Steps
    yLabel: Loss
    bins: 500

  # Accuracy as a separate chart
  - ctype: line
    title: Accuracy
    xKey: step
    yKey: eval.accuracy
    bins: 500

  # Latest rendered frames
  - ctype: image
    title: Rendered Frames
    glob: "frames/**/*.jpg"
    nCols: 4
    nRows: 2
    sort: date

  # Episode rollout videos
  - ctype: video
    title: Episode Rollouts
    glob: "videos/**/*.mp4"
    nCols: 2
    nRows: 2
    sort: name
```

---

**Next:** Learn how to compare multiple experiments with [Compare](compare.md).
