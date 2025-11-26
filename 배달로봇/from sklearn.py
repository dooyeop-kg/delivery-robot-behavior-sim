from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = ['path_length','displacement','straightness',
            'traj_avg_speed','stop_rate','direction_change_rate']

X = df[features]

# 1) Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) 클러스터 개수 찾기
import matplotlib.pyplot as plt

inertia=[]
K_range=range(2,8)

for k in K_range:
    km=KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()
