import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Data for with warmup
forward_with_warmup = [0.0325, 0.0308, 0.0285, 0.0282, 0.0304, 0.0308, 0.0307, 0.0299, 0.0310, 0.0307]
backward_with_warmup = [0.0534, 0.0474, 0.0455, 0.0496, 0.0469, 0.0507, 0.0482, 0.0483, 0.0506, 0.0496]

# Data for without warmup
forward_without_warmup = [0.4950, 0.0423, 0.0310, 0.0313, 0.0336, 0.0299, 0.0310, 0.0309, 0.0283, 0.0278]
backward_without_warmup = [0.2272, 0.0624, 0.0603, 0.0616, 0.0483, 0.0470, 0.0493, 0.0500, 0.0473, 0.0455]

# Create x-axis values
steps = np.arange(1, 11)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot forward passes (same color, different line styles)
plt.plot(steps, forward_with_warmup, 'o-', label='Forward (with warmup)', 
         color='#2E86C1', linewidth=2, markersize=8, alpha=0.8)
plt.plot(steps, forward_without_warmup, 'o--', label='Forward (no warmup)', 
         color='#2E86C1', linewidth=2, markersize=8, alpha=0.8)

# Plot backward passes (same color, different line styles)
plt.plot(steps, backward_with_warmup, 's-', label='Backward (with warmup)', 
         color='#E67E22', linewidth=2, markersize=8, alpha=0.8)
plt.plot(steps, backward_without_warmup, 's--', label='Backward (no warmup)', 
         color='#E67E22', linewidth=2, markersize=8, alpha=0.8)

# Customize the plot
plt.xlabel('Step Number', fontsize=12, fontweight='bold')
plt.ylabel('Time (ms)', fontsize=12, fontweight='bold')
plt.title('Comparison of Forward and Backward Pass Times\nWith and Without Warmup Steps', 
          fontsize=14, fontweight='bold', pad=20)

# Customize grid
plt.grid(True, linestyle='--', alpha=0.3)

# Customize legend
plt.legend(fontsize=10, frameon=True, framealpha=0.95, 
           edgecolor='gray', fancybox=False)

# Set y-axis limit to better show the difference
plt.ylim(0, 0.5)

# Customize ticks
plt.xticks(steps, fontsize=10)
plt.yticks(fontsize=10)

# Add some padding to the plot
plt.tight_layout()

# Save the plot with high DPI
plt.savefig('warmup_comparison.png', dpi=300, bbox_inches='tight')
plt.close() 