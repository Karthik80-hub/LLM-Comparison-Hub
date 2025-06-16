
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the evaluation file
df = pd.read_csv('ai_prompt_eval_template.csv')

# Drop incomplete rows
score_cols = ['helpfulness', 'correctness', 'coherence', 'tone_score']
df = df.dropna(subset=score_cols)
df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')

# Average scores per model
avg_scores = df.groupby('model')[score_cols].mean()
print("\nAverage Scores by Model:")
print(avg_scores)

# Plot average scores
plt.figure(figsize=(10, 6))
avg_scores.plot(kind='bar')
plt.title("Average Evaluation Scores by Model")
plt.ylabel("Average Score (1–5)")
plt.xlabel("Model")
plt.ylim(0, 5)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("model_avg_scores_chart.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df[score_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Correlation Between Evaluation Metrics")
plt.tight_layout()
plt.savefig("eval_score_correlation_heatmap.png")
plt.show()
