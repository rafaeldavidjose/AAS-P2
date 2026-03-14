import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10

# Create results directory if it doesn't exist
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Load the CSV files
print("Loading data...")
heuristic = pd.read_csv(os.path.join(OUT_DIR, "heuristic_metrics.csv"))
mcts = pd.read_csv(os.path.join(OUT_DIR, "mcts_metrics.csv"))
rl = pd.read_csv(os.path.join(OUT_DIR, "rl_metrics.csv"))

# ===================================================================
# FIGURE 1: Score Progression (Line Plot)
# ===================================================================
print("Generating score progression plot...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(heuristic['episode'], heuristic['score'], 
        'o-', linewidth=2, markersize=6, label='Heuristic', color='#2E86AB')
ax.plot(mcts['episode'], mcts['score'], 
        's-', linewidth=2, markersize=6, label='MCTS', color='#A23B72')
ax.plot(rl['episode'], rl['score'], 
        '^-', linewidth=2, markersize=6, label='RL (DQN)', color='#F18F01')

ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Score Progression Across Episodes', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.5, 10.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "score_progression.png"), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "score_progression.pdf"), bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/score_progression.png")
plt.close()

# ===================================================================
# FIGURE 2: Score Distribution (Box Plot)
# ===================================================================
print("Generating score distribution box plot...")
fig, ax = plt.subplots(figsize=(8, 6))

data = [heuristic['score'], mcts['score'], rl['score']]
labels = ['Heuristic', 'MCTS', 'RL (DQN)']
colors = ['#2E86AB', '#A23B72', '#F18F01']

bp = ax.boxplot(data, labels=labels, patch_artist=True,
                showmeans=True, meanline=True,
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='green', linewidth=2, linestyle='--'))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add legend for median and mean
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, label='Median'),
    Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Mean')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "score_distribution.pdf"), bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/score_distribution.png")
plt.close()

# ===================================================================
# FIGURE 3: Decision Time Comparison (Bar Chart)
# ===================================================================
print("Generating decision time comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

agents = ['Heuristic', 'MCTS', 'RL (DQN)']
avg_times = [
    heuristic['avg_decision_time_ms'].mean(),
    mcts['avg_decision_time_ms'].mean(),
    rl['avg_decision_time_ms'].mean()
]
max_times = [
    heuristic['max_decision_time_ms'].mean(),
    mcts['max_decision_time_ms'].mean(),
    rl['max_decision_time_ms'].mean()
]

x = np.arange(len(agents))
width = 0.35

bars1 = ax.bar(x - width/2, avg_times, width, label='Average Decision Time',
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, max_times, width, label='Max Decision Time',
               color=colors, alpha=0.5, edgecolor='black', linewidth=1.2, hatch='//')

ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
ax.set_ylabel('Decision Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('Decision Time Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(agents, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "decision_time_comparison.png"), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "decision_time_comparison.pdf"), bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/decision_time_comparison.png")
plt.close()

# ===================================================================
# FIGURE 4: Combined Summary Figure (2x2 grid)
# ===================================================================
print("Generating combined summary figure...")
fig = plt.figure(figsize=(14, 10))

# Score progression
ax1 = plt.subplot(2, 2, 1)
ax1.plot(heuristic['episode'], heuristic['score'], 'o-', linewidth=2, markersize=5, label='Heuristic', color='#2E86AB')
ax1.plot(mcts['episode'], mcts['score'], 's-', linewidth=2, markersize=5, label='MCTS', color='#A23B72')
ax1.plot(rl['episode'], rl['score'], '^-', linewidth=2, markersize=5, label='RL (DQN)', color='#F18F01')
ax1.set_xlabel('Episode', fontweight='bold')
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('(a) Score Progression', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')

# Score distribution
ax2 = plt.subplot(2, 2, 2)
bp = ax2.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('(b) Score Distribution', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# Decision time (log scale for better visibility)
ax3 = plt.subplot(2, 2, 3)
x_pos = np.arange(len(agents))
bars = ax3.bar(x_pos, avg_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax3.set_yscale('log')
ax3.set_xlabel('Agent', fontweight='bold')
ax3.set_ylabel('Avg Decision Time (ms, log scale)', fontweight='bold')
ax3.set_title('(c) Decision Time (Log Scale)', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(agents, fontsize=10)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# Steps per episode
ax4 = plt.subplot(2, 2, 4)
ax4.plot(heuristic['episode'], heuristic['steps'], 'o-', linewidth=2, markersize=5, label='Heuristic', color='#2E86AB')
ax4.plot(mcts['episode'], mcts['steps'], 's-', linewidth=2, markersize=5, label='MCTS', color='#A23B72')
ax4.plot(rl['episode'], rl['steps'], '^-', linewidth=2, markersize=5, label='RL (DQN)', color='#F18F01')
ax4.set_xlabel('Episode', fontweight='bold')
ax4.set_ylabel('Steps', fontweight='bold')
ax4.set_title('(d) Episode Length', fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "combined_summary.png"), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "combined_summary.pdf"), bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/combined_summary.png")
plt.close()

# ===================================================================
# Print Summary Statistics
# ===================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for name, df in [("Heuristic", heuristic), ("MCTS", mcts), ("RL (DQN)", rl)]:
    print(f"\n{name} Agent:")
    print(f"  Score:    Mean={df['score'].mean():.1f}, Std={df['score'].std():.1f}, Best={df['score'].max()}")
    print(f"  Steps:    Mean={df['steps'].mean():.1f}, Std={df['steps'].std():.1f}")
    print(f"  Dec Time: Mean={df['avg_decision_time_ms'].mean():.3f}ms, Max={df['max_decision_time_ms'].mean():.3f}ms")
    print(f"  Memory:   Mean={df['peak_memory_mb'].mean():.2f}MB")

print("\n" + "="*70)
print("All plots generated successfully!")
print("="*70)