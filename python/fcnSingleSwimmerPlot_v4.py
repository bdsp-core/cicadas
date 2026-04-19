# fcnSingleSwimmerPlot_v4.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox

def fcnSingleSwimmerPlot_v4(T: pd.DataFrame,
                            max_time: float = 168.0,
                            time_resolution: int = 500) -> plt.Figure:
    """
    Create swimmer plots for L(t) and A(t) with heatmap-like coloring.

    Args:
        T: DataFrame with columns at least:
           ['sid','t','Rx','L','A','Y'] and optionally 'V'
        max_time: right edge of time horizon (hours)
        time_resolution: number of horizontal samples for the heatmaps

    Returns:
        Matplotlib Figure object.
    """
    required_cols = {'sid','t','Rx','L','A','Y'}
    missing = required_cols - set(T.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {sorted(missing)}")

    # Prep
    subject_ids = np.array(sorted(T['sid'].unique()))
    n_subjects = subject_ids.size
    has_V = 'V' in T.columns

    # Per-subject summaries (max time, initial Rx, died, censored, event time)
    max_times = np.zeros(n_subjects)
    initial_treatment = np.zeros(n_subjects, dtype=bool)
    died = np.zeros(n_subjects, dtype=bool)
    censored = np.zeros(n_subjects, dtype=bool)
    event_times = np.zeros(n_subjects)

    for i, pid in enumerate(subject_ids):
        pdata = T.loc[T['sid'] == pid].sort_values('t')
        tvals = pdata['t'].to_numpy()
        max_times[i] = float(tvals.max()) if len(tvals) else 0.0
        initial_treatment[i] = (int(pdata['Rx'].iloc[0]) == 1)

        # death / censor flags
        died[i] = bool((pdata['Y'] > 0).any())
        if has_V:
            censored[i] = bool((pdata['V'] > 0).any())
            ev_idx = np.where((pdata['Y'] > 0) | (pdata['V'] > 0))[0]
        else:
            ev_idx = np.where(pdata['Y'] > 0)[0]

        if ev_idx.size > 0:
            event_times[i] = float(pdata['t'].iloc[ev_idx[0]])
        else:
            event_times[i] = max_times[i]

    # Define groups
    completed = (~died) & (~censored)
    g1 = initial_treatment & died
    g2 = initial_treatment & censored
    g3 = initial_treatment & completed
    g4 = (~initial_treatment) & died
    g5 = (~initial_treatment) & censored
    g6 = (~initial_treatment) & completed

    # IDs and times per group
    def _group_ids_times(mask, use_event=True):
        ids = subject_ids[mask]
        times = (event_times if use_event else max_times)[mask]
        order = np.argsort(times, kind='mergesort')
        return ids[order]

    group1_ids = _group_ids_times(g1, use_event=True)
    group2_ids = _group_ids_times(g2, use_event=True)
    group3_ids = _group_ids_times(g3, use_event=False)
    group4_ids = _group_ids_times(g4, use_event=True)
    group5_ids = _group_ids_times(g5, use_event=True)
    group6_ids = _group_ids_times(g6, use_event=False)

    sorted_subject_ids = np.concatenate([
        group1_ids, group2_ids, group3_ids,
        group4_ids, group5_ids, group6_ids
    ])

    # One divider between initially treated (g1,g2,g3) and untreated (g4,g5,g6)
    group_divider = len(group1_ids) + len(group2_ids) + len(group3_ids)

    # Time grid for heatmaps
    time_grid = np.linspace(0.0, max_time, time_resolution)

    # Colors & normalization helpers
    max_L = float(np.nanmax(T['L'])) if len(T) else 1.0
    max_A = float(np.nanmax(T['A'])) if len(T) else 1.0
    max_L = max(max_L, 1e-12)
    max_A = max(max_A, 1e-12)

    # Build heatmap arrays (RGB)
    L_image = np.ones((n_subjects, time_resolution, 3), dtype=float)  # default white
    A_image = np.ones((n_subjects, time_resolution, 3), dtype=float)

    # Map intensity -> color (L: light to dark red, A: light to dark blue)
    # L: light red [1,0.8,0.8] -> dark red [0.5,0,0]
    # A: light blue [0.8,0.8,1] -> dark blue [0,0,0.5]
    def color_from_L(intensity):  # intensity in [0,1]
        r = 1.0 - 0.5*intensity
        g = 0.8*(1.0 - intensity)
        b = 0.8*(1.0 - intensity)
        return r, g, b

    def color_from_A(intensity):
        r = 0.8*(1.0 - intensity)
        g = 0.8*(1.0 - intensity)
        b = 1.0 - 0.5*intensity
        return r, g, b

    # Vectorized per-subject filling
    sid_to_row = {sid: i for i, sid in enumerate(sorted_subject_ids)}
    for pid in sorted_subject_ids:
        row = sid_to_row[pid]
        pdata = T.loc[T['sid'] == pid].sort_values('t')
        times = pdata['t'].to_numpy()
        Lvals = pdata['L'].to_numpy()
        Avals = pdata['A'].to_numpy()
        Yvals = pdata['Y'].to_numpy()
        Vvals = pdata['V'].to_numpy() if has_V else np.zeros_like(Yvals)

        # Event time & type
        ev_time = float(times.max()) if times.size else 0.0
        event_is_death = False
        event_is_censored = False
        if has_V:
            ev_idx = np.where((Yvals > 0) | (Vvals > 0))[0]
            if ev_idx.size > 0:
                ev_time = float(times[ev_idx[0]])
                event_is_death = bool(Yvals[ev_idx[0]] > 0)
                event_is_censored = bool(Vvals[ev_idx[0]] > 0)
        else:
            ev_idx = np.where(Yvals > 0)[0]
            if ev_idx.size > 0:
                ev_time = float(times[ev_idx[0]])
                event_is_death = True

        # For each t in time_grid, use last observed value at time <= t
        # idxs = rightmost index <= t
        if times.size:
            idxs = np.searchsorted(times, time_grid, side='right') - 1
            idxs = np.clip(idxs, 0, times.size - 1)
            Lg = Lvals[idxs]
            Ag = Avals[idxs]
        else:
            Lg = np.zeros_like(time_grid)
            Ag = np.zeros_like(time_grid)

        # After event -> black if death, light gray if censored
        post_mask = time_grid > ev_time

        # Pre-event color mapping (sqrt scaling)
        L_int = np.sqrt(np.clip(Lg / max_L, 0.0, 1.0))
        A_int = np.sqrt(np.clip(Ag / max_A, 0.0, 1.0))

        # Fill L row
        rL, gL, bL = color_from_L(L_int)
        L_rgb = np.stack([rL, gL, bL], axis=-1)
        if np.any(post_mask):
            if event_is_death:
                L_rgb[post_mask, :] = np.array([0.0, 0.0, 0.0])  # black
            elif event_is_censored:
                L_rgb[post_mask, :] = np.array([0.7, 0.7, 0.7])  # light gray
        L_image[row, :, :] = L_rgb

        # Fill A row
        rA, gA, bA = color_from_A(A_int)
        A_rgb = np.stack([rA, gA, bA], axis=-1)
        if np.any(post_mask):
            if event_is_death:
                A_rgb[post_mask, :] = np.array([0.0, 0.0, 0.0])
            elif event_is_censored:
                A_rgb[post_mask, :] = np.array([0.7, 0.7, 0.7])
        A_image[row, :, :] = A_rgb

    # Build colormaps for colorbars (256 steps, with the same sqrt mapping)
    def make_cmap(make_color_fn):
        steps = 256
        vals = np.linspace(0, 1, steps)
        vals = np.sqrt(vals)
        colors = np.array([make_color_fn(v) for v in vals])
        return ListedColormap(colors)

    cmap_L = make_cmap(color_from_L)
    cmap_A = make_cmap(color_from_A)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # Display images (first row at top, like MATLAB with YDir reverse)
    extent = [0.0, max_time, 1, n_subjects]
    ax1.imshow(L_image, aspect='auto', origin='upper', extent=extent)
    ax2.imshow(A_image, aspect='auto', origin='upper', extent=extent)

    # Add colorbars (using ScalarMappable so the bar reflects value→color)
    cbar1 = fig.colorbar(
        ScalarMappable(cmap=cmap_L, norm=plt.Normalize(0, max_L)),
        ax=ax1
    )
    cbar1.set_label('Disease Intensity L(t)', fontsize=11)

    cbar2 = fig.colorbar(
        ScalarMappable(cmap=cmap_A, norm=plt.Normalize(0, max_A)),
        ax=ax2
    )
    cbar2.set_label('Treatment Intensity A(t)', fontsize=11)

    # Group divider line
    if 0 < group_divider < n_subjects:
        yline = group_divider + 0.5
        ax1.plot([0, max_time], [yline, yline], 'k-', linewidth=2)
        ax2.plot([0, max_time], [yline, yline], 'k-', linewidth=2)

    # Axis cosmetics
    for ax in (ax1, ax2):
        ax.set_xlim(0, max_time)
        ax.set_ylim(0.5, n_subjects + 0.5)
        ax.set_yticks([])          # no subject IDs on y-axis
        ax.set_yticklabels([])
        ax.grid(True, axis='x')

    # Title/labels
    ax1.set_title('Swimmer Plot: Disease Intensity L(t)', fontsize=12)
    ax2.set_title('Swimmer Plot: Treatment Intensity A(t)', fontsize=12)
    ax2.set_xlabel('Time (hours)', fontsize=11)

    # Figure background
    fig.patch.set_facecolor('white')

    # Manual layout to mimic MATLAB’s “full width + tight vertical spacing”
    vertical_gap = 0.015   # gap between subplots
    left_margin  = 0.06
    right_margin = 0.10    # extra space for colorbars
    top_margin   = 0.02
    bottom_margin = 0.20   # reserve room if you later add a survival plot

    width = 1 - left_margin - right_margin
    total_height = 1 - top_margin - bottom_margin
    subplot_height = (total_height * 2/3 - vertical_gap) / 2  # two swimmer plots

    # Positions in figure-normalized coords: [left, bottom, width, height]
    ax1.set_position(Bbox.from_bounds(left_margin,
                                      1 - top_margin - subplot_height,
                                      width,
                                      subplot_height))
    ax2.set_position(Bbox.from_bounds(left_margin,
                                      1 - top_margin - 2*subplot_height - vertical_gap,
                                      width,
                                      subplot_height))

    return fig
