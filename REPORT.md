What domain did you choose and why? What makes your dataset interesting?

I chose online gaming because companies often use player grouping to understand behavior and improve games. The features include time spent playing, skill level, spending, and social activity. These do not always move together, so the dataset becomes realistic and suitable for clustering.


How did you generate the data? How many natural clusters did you embed? How did you inject noise?

The dataset was created using three player types. Each feature was generated using a normal distribution around a mean value with some variation to make it realistic. Values were also clipped to avoid unrealistic results, and all groups were combined and shuffled. Noise was added by allowing variation in each feature using standard deviation in numpy.random.normal. This creates realistic overlap between players so the clusters are not perfectly separated. After generating the data, values were also clipped to keep them within realistic limits, and all clusters were combined and shuffled to remove any ordering pattern.


What was your optimal K? Did the Elbow Method and Silhouette Score agree? 

The best value of K was 3. Both the elbow method and silhouette score agreed on this. The elbow graph clearly bends at 3, and the silhouette score is highest at 3 with around 0.60. After that, performance decreases, so K=3 is the correct choice.


Name each cluster and describe what makes it unique in terms of the original features.

Hardcore Grinders play the most hours and have the highest skill but spend little money and are not very social. Social Casuals play the least, have low skill, low spending, but very high social interaction. Whale Competitors have moderate playtime and skill but spend the most money by a large margin. Each group is mainly defined by one strong feature.


What happened when you removed feature scaling ? How did the clusters change? Show a before/after comparison

Without scaling, spending dominated the clustering because its values are much larger than other features. The model grouped players mostly by spending and ignored other behavior. Even if scores looked slightly better, the clusters were not meaningful. With scaling, all features contributed fairly, making results more balanced.


Were there any points with negative silhouette scores? What does that mean about those specific data points?

There were no negative silhouette scores in this dataset. Most values were positive, showing that points are mostly assigned to correct clusters. A negative score would mean a point fits better in another cluster, which did not happen here.



What are the limitations of K-Means for your specific dataset? When might it fail?

K Means needs the number of clusters in advance and assumes clusters are round-shaped, which may not match real data. It is also sensitive to outliers and forces every point into one group, even if it is between two types.so one player spending 500 dollars a month could change or sidrupt an entire centroid. 


If you could add a 4th feature, what would it be and why? How might it change the clustering?
Win rate would be a good extra feature because it shows team success, not just individual skill. It could separate players who are strong alone from those who perform better in teams, possibly revealing more detailed clusters.
