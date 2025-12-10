# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# -----------------------------
# 2D Scatter Grid (Pairplot)
# -----------------------------
sns.set(style="ticks", palette="bright")
pairplot = sns.pairplot(df, hue="species", markers=["o", "s", "D"], diag_kind="hist")
pairplot.fig.suptitle("Iris Dataset Pairplot (2D Scatter Grid)", y=1.02)
plt.show()  # Show the 2D pairplot first

# -----------------------------
# 3D Scatter Plot
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter points for each species
colors = ['r', 'g', 'b']
for color, species in zip(colors, iris.target_names):
    subset = df[df['species'] == species]
    ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
               subset['petal length (cm)'], label=species, color=color, s=50)

ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
ax.set_title('Iris Dataset 3D Scatter Plot')
ax.legend()

plt.show()  # Show the 3D scatter plot
