import pandas
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

dataset = pandas.read_csv('dataset.csv')

# print(dataset)

dataset = dataset.values 
#convert dataframe into a matrix

# print(dataset)


pyplot.scatter(dataset[:,0],dataset[:,1])
pyplot.savefig('scatterplot.png')
pyplot.close()


def run_kmeans(n,dataset):

	machine = KMeans(n_clusters=n)  # or we can use KMeans
	machine.fit(dataset)
	results = machine.predict(dataset)
	centroids = machine.cluster_centers_ # center of the category
	ssd = machine.inertia_ # distance from the center of the category to all points in the category
	if n>1:
		silhouette = silhouette_score(dataset, machine.labels_, metric='euclidean')
	else:
		silhouette = 0
	pyplot.scatter(dataset[:,0],dataset[:,1],c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s = 200)
	pyplot.savefig('scatterplot_kmean_' + str(n) + '.png')
	pyplot.close()
	return ssd, silhouette


# silhouette score = (b-a)/max(b,a) # b is the avg distance between the clusters centers; a is the avg distance within the clusters, check the explanation of this


# print(run_kmeans(1,dataset))
# print(run_kmeans(2,dataset))
# print(run_kmeans(3,dataset))
# print(run_kmeans(4,dataset))
# print(run_kmeans(5,dataset))
# print(run_kmeans(6,dataset))
# print(run_kmeans(7,dataset))


result = [run_kmeans(i+1,dataset) for i in range(7)]
print(result)

ssd_result = [ i[0] for i in result]

pyplot.plot(range(1,8), result)
pyplot.savefig("kmeans_ssd.png")
pyplot.close()

silhouette_result = [ i[1] for i in result][1:]

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("kmeans_silhouette.png")
pyplot.close()

print(silhouette_result.index(max(silhouette_result))+2)

# kmedoids(median) is less affected by outliers, kmeans(mean)

# implement kmean and find optimal n clusters

# watch the recording he uploaded and this one(11.03) before the class on next Thursday




