# https://shiny.rstudio.com/reference/shiny/0.14/withMathJax.html
# https://shiny.rstudio.com/gallery/mathjax.html


library(shiny)
library(datasets)
library(factoextra)
library(mclust)

# Define server logic required to draw a histogram
shinyServer(function(input, output, session) {
  data(iris)
  
#-----------------------------------------------------------------------------------------------------------------------------------------------------------  
   
   
    introduction_text <- HTML(paste0("One of the most important problems in data mining is to extract meaningful information and relationships among data objects that are unlabeled, latent, or even previously unseen.  In their simplest form, these problems intend to identify if certain data objects belong to groups or clusters that indicate some aspect of similarity.  For example, we might be interested in identifying if a set of books or articles are written on the same topic, or we might be interested in understanding if a set of customers have similar interests based on their purchasing patterns.  These problems are collectively known as clustering in the field of data mining.", "<b>"," Clustering", "</b>" , " is a technique to group similar items in to subsets or clusters"))
    introduction_text2 <- HTML(paste0("The goal of this tutorial is to provide readers with a overview of different clustering algorithms used widely in data mining.  We will first describe the problem of clustering and then classify existing clustering algorithms into broad classes (parametric and non-parametric), and then describe each of one of them in detail.  This tutorial will provide both mathematical foundations for understanding these algorithms and modeling techniques used for clustering.  We will use in-built data sets in R for all our examples and will show how different algorithms work using the interactive shiny app."))
    introduction_text3 <- HTML(paste0("Clustering is a hard problem because, unlike several other problems in data mining, we are not working with supervised data that already have true labels associated with them.  We are only trying to identify objects that show patterns of similarity or dissimilarity and group them together.  In some cases, we may assume the conditional probability distributions of all or some subset of the data objects with respect to some parameters.  For example, we may model the data as Gaussian Mixture Models and then apply the Expectation-Maximization (EM) algorithm to cluster the data based on the  parameters of the mixture models (mixture probabilities, means, and variances).  These are commonly referred to as parametric clustering problems."))
    introduction_text4 <- HTML(paste0("Assuming a parametric distribution is not always possible.  These are cases where we know little to nothing about how the data was obtained and we are trying to just group them together based on minimizing some similarity criterion.  These are commonly referred to as non-parametric clustering problems.  There are many established ways in which one can compute similarity.  The most common approach is to minimize a similarity function or distances (e.g., Euclidean distances between two X data vectors).  The widely used k-means clustering algorithm is an example of this.  Similarity can also be computed based on connections or linkage between the data objects.  Hierarchical clustering algorithms make use of this strategy."))
    introduction_text5 <- HTML(paste0("Thus, there are two views of clustering: one that partitions data into clusters based on similarity, and the other that partitions data such that each object in a cluster belongs to the same distribution.  Over the next few sections, we will introduce each of these clustering algorithms and discuss them in detail."))
    
    output$intro_text <- renderText({HTML(paste0("<h1>","Introduction","</h1>",
                                                  "<p>", "<font face='times new roman', size='4.5'>", introduction_text,"</font>",  "</p>"))})
    output$Intro <- renderImage({
      filename <- normalizePath(file.path(paste('intro', '.png', sep='')))
      list(src = filename,width=860,height=300)
     }, deleteFile = FALSE)
    output$intro_text1 <- renderText({HTML(paste0("<p>", "<font face='times new roman', size='4.5'>", introduction_text2, "</font>", "</p>",
                                                  "<p>", "<font face='times new roman', size='4.5'>", introduction_text3, "</font>", "</p>",
                                                  "<p>", "<font face='times new roman', size='4.5'>", introduction_text4, "</font>", "</p>",
                                                  "<p>", "<font face='times new roman', size='4.5'>", introduction_text5, "</font>", "</p>"))})
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------  
    
    parametric_text1 <- HTML(paste0("Parametric clustering deals with data coming in from distinct clusters that each contain data objects 
    belonging to the same parametric probability distribution.  Therefore, we can view this as a parameter estimation problem, 
    where we initialize the parameters to random values and then estimate the most likely values.  The common practice is to model 
    the data as a mixture model, where each of the individual components of the mixture have different parameters and therefore 
    represent a particular cluster.  The intuition behind this is that is similar data objects are most likely correlated 
    and belong to the same probability distribution.  Since we are trying to find the most likely values for the parameters for 
    these distributions, they can be iteratively estimated using the EM algorithm.  The following equation gives the general form 
    of a mixture model."))
    
    parametric_text2 <- HTML(paste0("Here, \\(\\pi_k\\) represents the mixing probability for the data objects in the \\(k^{th}\\) 
                                    cluster, and \\(f(x, \\theta_k)\\) is a parametric probability density/mass function with 
                                    the parameter \\(\\theta_k\\), representing the \\(k^{th}\\) cluster.  These parameters are initialized 
                                    in the E-step and are iteratively estimated in the M-step, until convergence to a local/global optima."))
    
    parametric_text3 <- HTML(paste0("The most common mixture model used is a Gaussian Mixture Model (GMM) where the parameters 
                                    of the mixture model are the mixing probabilities (πk), the means (μk), and the variances 
                                    (σk). This is convenient because it allows our data to be real numbers (or a vector of real 
                                    numbers). The general form of such a GMM (univariate in this case) is given by:"))
    parametric_text4 <- HTML(paste0("Here, N(x;μ ,σ ) = 1 e is the parametric density function, given by a
   univariate Gaussian, with parameters μk and σk, representing the kth cluster. For simplicity, we will first describe 
   how to estimate the parameters of a univariate gaussian and then extend it for the multivariate case."))
    
    parametric_text5 <- HTML(paste0("Since these parameters need to be estimated to their most likely values, this resembles 
                                    the problem of finding the maximum likelihood estimate when some of our variables are 
                                    hidden/latent. The hidden variables in our case are the cluster labels zi which indicate 
                                    if xi belongs to cluster k, and the parameter πk can now be written as P(zi =k).", "<br>", 
                                    "Therefore, the log-likelihood of the complete data can be given by:"))
    
    parametric_text6 <- HTML(paste0("We use the EM algorithm to find the most likely estimates for these parameters.  
                                    In the E-step (expectation), we choose the number of clusters k and then initialize 
                                    the parameters \\(\\pi_k\\), \\(\\mu_k\\), and \\(\\sigma_{k}\\) to either random values or careful guesses
                                    (for example, we can choose both \\(\\mu_k\\) to 0 and \\(\\sigma_k\\) to 0, and rely on the M-step 
                                    for convergence to most likely values for these).  This will allow us to compute the expected
                                    values of the posterior (also known as membership weight \\(w_{ik}\\) using the following
                                    equation."))
    output$intro_text2 <- renderText({HTML(paste0("<h1>","Parametric Clustering","</h1>","<p>", "<font face='times new roman', size='4.5'>", parametric_text1,"</font>",  "</p>"))})
   
     # use of withMathJax() function from the link:
    #https://stackoverflow.com/questions/30169101/latex-formula-in-shiny-panel
     
    output$formula <- renderUI({
      withMathJax(paste0("$$f(x) = \\sum^{K}_{k=1}\\pi_k f(x; \\theta_k) $$"))
    })
    
    output$formula1 <- renderUI({
      withMathJax(paste0("$$f(x) = \\pi_1 f(x; \\theta_1) + \\pi_2 f(x; \\theta_2) + . . . + \\pi_K f(x; \\theta_K) $$"))
    })
    
    output$intro_text3 <- renderText({HTML(paste0("<p>", "<font face='times new roman', size='4.5'>", parametric_text2,"</font>",  "</p>",
                                                  "<p>", "<font face='times new roman', size='4.5'>", parametric_text3,"</font>",  "</p>"))})
    
    output$formula2 <- renderUI({
      withMathJax(paste0("$$f(x) = \\sum^{K}_{k=1}\\pi_k \\mathcal{N}(x; \\mu_k, \\sigma_k) $$"))
    })
    
    output$formula3 <- renderUI({
      withMathJax(paste0("$$f(x) = \\pi_1 \\mathcal{N}(x; \\mu_1, \\sigma_1) + \\pi_2 \\mathcal{N}(x; \\mu_2, \\sigma_2) + . . . + \\pi_K \\mathcal{N}(x; \\mu_K, \\sigma_K $$"))
    })
    
    output$intro_text4 <- renderText({HTML(paste0("<p>", "<font face='times new roman', size='4.5'>", parametric_text4,"</font>",  "</p>",
                                                  "<p>", "<font face='times new roman', size='4.5'>", parametric_text5,"</font>",  "</p>"))})
    
    
    
    output$formula4 <- renderUI({
      withMathJax(paste0("$$log(P(x, z | \\mu, \\sigma, \\pi)) =  \\sum_{i=1}^{n}\\sum_{k=1}^{K}I(z_i=k)(log(\\pi_k) + 
      log(\\mathcal{N}(x_i; \\mu_k, \\sigma_k)))$$"))
    })
    
    output$intro_text5 <- renderText({HTML(paste0("<p>", "<font face='times new roman', size='4.5'>", parametric_text6,"</font>",  "</p>"))})
                                                  
    output$formula5 <- renderUI({
      withMathJax(HTML(paste0("$$w_{ik} = P(z_{ik} = 1 | x_i, \\mu_k, \\sigma_k, \\pi_k)) =  \\dfrac{\\pi_k P(x_i | z_k, \\mu_k, \\sigma_k, 
                         \\pi_k)}{\\sum_{k=1}^{K}\\pi_k P(x_i | z_k, \\mu_k, \\sigma_k, \\pi_k)}$$", "<br>", 
                         "In the M-step, we update \\(\\pi_k\\), \\(\\mu_k\\), and \\(\\sigma_k\\) as follows:", 
                         "$$\\pi_k^* \\longleftarrow \\frac{\\sum_{i=1}^{n}w_{ik}}{n}$$", 
                         "$$\\mu_k^* \\longleftarrow \\frac{\\sum_{i=1}^{n}w_{ik}x_i}{\\sum_{i=1}^{n}w_{ik}}$$",
                         "$$\\sigma^2_k * \\longleftarrow \\frac{\\sum_{i=1}^{n}w_{ik}(x_i-\\mu_i)^2}{\\sum_{i=1}^{n}w_{ik}^2}$$", "<br>",
                         "In case of a multivariate gaussian, the update rules will be as follows:", "<br>", 
                         "$$\\pi_k^* \\longleftarrow \\frac{\\sum_{i=1}^{n}w_{ik}}{n}$$", 
                         "$$\\vec{\\mu}^{\\,*}_{k} \\longleftarrow \\frac{\\sum_{i=1}^{n}w_{ik}\\vec{x}_i}{\\sum_{i=1}^{n}w_{ik}}$$",
                         "$$\\sigma_k^* \\longleftarrow \\frac{\\sum_{i=1}^{n}w_{ik}(\\vec{x}_i-\\vec{\\mu}_i)(\\vec{x}_i-\\vec{\\mu}_i)^T}{\\sum_{i=1}^{n}w_{ik}}$$",
                         "<p>", "<font face='times new roman', size='4.5'>",
                         "To illustrate this, we randomly generated two different normal distributions of data with means (20, 40) and
                         variance (5, 2). We want see if the gaussian mixture model is detecting these two different distributions and assigning
                         them as two different clusters."," From the graph below we can say that the gaussian mixture model clusters data points
                         that belong to the same distribution.","</font>",  "</p>")))
    })
    
    
    x_data <- c(rnorm(100, 20,5), rnorm(100, 40, 2 ))
    y_data <- rnorm(400, 10, 3)
    df <- data.frame(x_data, y_data)
    cluster_gmm <- Mclust(df)
    plot_gmm <- plot(cluster_gmm, what = "classification")
    
    
    output$gmm <- renderPlot(plot(cluster_gmm, what = "classification" ), height = 400, width = 500)
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------  
    
    nonparametric_text1 <- HTML(paste0("In this section, we will discuss clustering algorithms that do not assume that the data 
    belongs to a certain mix of parametric distributions.  These algorithms look for patterns of similarity or dissimilarity 
    instead.  We will discuss three non-parametric clustering algorithms: ","<i>","k-means","</i>"," which is a distance-based algorithm, 
    ","<i>","DBSCAN","</i>"," which is ","density-based clustering"," that clusters data objects based on how densely they are populated in space, and 
    ","<i>","hierarchical clustering","</i>"," that groups objects based on connections or linkage."))
    
    nonparametric_text2 <- HTML(paste0("One of the most common distance-based clustering algorithms is K-means clustering.  
                                       The intuition behind k-means clustering is that each cluster has a representative data 
                                       point which lies at the centroid of that cluster, and all data points that belong to that 
                                       cluster lie in close proximity to that centroid.  There can be several measures of 
                                       proximity, such as cosine similarity, Jaccard's distance, and Euclidean distance.  
                                       But regardless of the measure, it is important to find clusters such that the internal 
                                       distance between the data points within a cluster is small, and the external distance 
                                       between members of different clusters is large enough."))
    nonparametric_text3 <- HTML(paste0("The first (initialization) step in k-means clustering is to choose the number of clusters 
                                       K and the initial locations of their centroids.  Ideally, these centroids should be as far 
                                       away from each other as possible to avoid overlap.  The measure of proximity can be decided
                                       based on the problem being considered."))
    nonparametric_text4 <- withMathJax(HTML(paste0("We next iteratively assign each data point to the nearest centroid.  
                                       For example, if our measure of similarity is Euclidean distance, the cluster label 
                                                   \\(z_i\\) for the data point \\(x_i\\) would be given by:")))
    
    output$intro_text11 <- renderText({HTML(paste0("<h1>","Non Parametric Clustering","</h1>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text1,"</font>",  "</p>",
                                                   "<h2>","Distance Based clustering","</h2>",
                                                   "<h3>","K-means clustering","</h3>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text2,"</font>",  "</p>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text3,"</font>",  "</p>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text4,"</font>",  "</p>"))})
    
    
    output$formula6 <- renderUI({
      withMathJax(paste0("$$z_i = \\underset{1\\leq k \\leq K}{\\arg\\min}\\lVert(x_i-c_k)\\rVert^2$$"))
    })
    
    nonparametric_text5 <- HTML(paste0("where \\(c_k\\) is the centroid of the \\(k^{th}\\) cluster.  Once all data points have been 
                                       assigned cluster labels, the centroids are re-computed by selecting the point in the 
                                       cluster that has the smallest distance to all other points in the cluster.  This process 
                                       is repeated iteratively until convergence."))
    
    nonparametric_text6 <- HTML(paste0("The interactive plot demonstrates the k-means clustering for Iris data set for different number of cluster (k) by changing the k value on the slider input."))
    
    output$intro_text12 <- renderText({HTML(paste0("<p>", "<font face='times new roman', size='4.5'>", nonparametric_text5,"</font>",  "</p>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text6,"</font>",  "</p>"))})
    
    
    
    data_iris = as.matrix(sapply(iris[,1:4], as.numeric))
    
    clusters_kmeans <- reactive({
      kmeans(data_iris, input$k)
    })
    
    
    # use of fviz_cluster function to visualize plots.
    #https://www.rdocumentation.org/packages/factoextra/versions/1.0.5/topics/fviz_cluster
    
    output$cluster_plot_kmeans <- renderPlot({
    fviz_cluster(clusters_kmeans(), data = data_iris, main = "K-Means", ellipse = TRUE, geom = "point", ggtheme = theme_minimal())
    
    })
    
    
    
    nonparametric_text7 <- HTML(paste0("Although k-means is a very simple algorithm, the results are very much dependent on the 
    initial selection of the number of clusters K, as you can see from the above illustration.  Therefore, it is always important to try k-means clustering with different 
    values of K and then choose the best one that minimizes some kind of a loss function (e.g., look for the elbow in the plot 
    of mean-squared error vs. K).  Also, k-means only finds local solutions.  Therefore, it is important to run k-means with 
    different initializations of the centroids.  The other major weakness of k-means is that it only chooses spherical clusters 
    that are balanced (of equal size).  This is because there isn't a well-defined mechanism to choose clusters of different sizes,
    and the measure of proximity used is typically Euclidean distance."))
    
    nonparametric_text8 <- HTML(paste0("In this section, we will discuss another distance-based clustering technique called 
    hierarchical clustering.  The difference here is that we are building a hierarchy of clusters based on a similarity or 
    distance metric.  Just like k-means clustering, we group data objects that lie close to each other by drawing connections 
    between them.  The longer the connection, the less likely are the two objects to lie in the same cluster.  Using this general
    principle, we can build a hierarchy of clusters in one of the two following ways."))
    
    nonparametric_text9 <- HTML(paste0("1. Agglomerative clustering, which is a bottom-up approach, where we first assign every 
    data object to its own cluster, and then start merging clusters based on how long or how short the inter-cluster connections
    are."))
    
    nonparametric_text10 <- HTML(paste0("2. Divisive clustering, which is top-down approach, where we first assign all data objects
                                       to just one cluster, and then start splitting clusters, again based on how long or how 
                                       short the connection between different objects in the cluster are."))
    nonparametric_text11 <- HTML(paste0("Both strategies can be easily implemented with the help of a dendrogram representation 
    of the clusters being created.  The root of the dendrogram tree represents a single large cluster with all data points and the
    leaves represent clusters that hold only one data point.  Therefore, moving up the dendrogram while merging clusters is 
    agglomerative clustering and moving down the dendrogram while splitting clusters is divisive clustering.  
    Splitting and merging of clusters depends on a cut-off distance chosen on the y-axis of the dendrogram.  
    The following figure shows an example."))
    
    output$intro_text13 <- renderText({HTML(paste0("<p>", "<font face='times new roman', size='4.5'>", nonparametric_text7,"</font>",  "</p>",
                                                   "<h3>","Hierarchical Clustering","</h3>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text8,"</font>",  "</p>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text9,"</font>",  "</p>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text10,"</font>",  "</p>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", nonparametric_text11,"</font>",  "</p>"))})
    output$cluster_plot_hierdend <- renderPlot({
      fviz_dend(clusters_hier(), data = data_iris, main  = "Dendogram", show_labels = FALSE, ellipse = TRUE, geom = "point", ggtheme = theme_minimal())
      
    })
    
    clusters_hier <- reactive({
      hcut(data_iris, input$k1, hc_method = "complete")
    })
    
    output$cluster_plot_hier <- renderPlot({
      fviz_cluster(clusters_hier(), data = data_iris, main  = "Hierarchical", ellipse = TRUE, geom = "point", ggtheme = theme_minimal())
      
    })
    nonparametric_text12 <- HTML(paste0("There are several ways to measure distances.  Single linkage (also called nearest neighbor) measures the smallest distance (e.g., Euclidean distance, cosine distance, etc.) between data objects in 
                                       two clusters.  This is useful for the agglomerative clustering approach.  Complete linkage (also called farthest neighbor) 
                                       measures the largest distance between data objects in two clusters. This is useful for the divisive learning approach.  
                                       We could also use the average distance between objects in two clusters, or use centroid distance like k-means clustering.  A 
                                       notable distance measure is the Ward's linkage which measures the increase in a loss function (variance or sum of squares) for the clusters being merged.  The choice of these
     measures can be very important and it varies based on the problem at hand."))
    output$intro_text14 <- renderText({HTML(paste0("<p>", "<font face='times new roman', size='4.5'>", nonparametric_text12,"</font>",  "</p>"))})
    
    output$link <- renderImage({
      filename <- normalizePath(file.path(paste('sca', '.png', sep='')))
      list(src = filename,width=860,height=500)
    }, deleteFile = FALSE)
#__________________________________________________________________________________________________________________________________________________________________________________
    
    density_text_1 <- HTML(paste0("DBSCAN stands for Density Based Spatial Clustering of Applications with Noise. 
    Basic idea behind DBSCAN is to group together observations of high density and find outlier points or noisy points 
    that are from low density regions. Unlike K-means, we need not provide the number of clusters for DBSCAN. But it takes two 
    input parameters 1. radius of the neighbourhood (r) and 2. minimum number of neighbourhood points within their radii(n).
An observation in the data set is considered as ","<i>"," Core Object","</i>"," if there are at least n other observations within 'r' radius
of the observation. These core observations form the dense part of the cluster. ","<i>"," Border objects ","</i>"," or edge objects are the 
  observations from the lower density region but are in the 'r'  neighbourhood of the core object. 
  ","<i>"," Noise object ","</i>","  is any observation which is neither a core object nor an edge object. Before going into clustering algorithm,
                                  we need to understand few concepts of Density Based Clustering like Reachability and Connectivity"))
    
    density_text_2 <- HTML(paste0(" 
A point x is reachable from a point y, if y is a core point and x is in 'r' neighborhood of y or there exists a chain of points between x, y say - (P1, P2, .....,Pn) such that (Pi+1) is directly density reachable from (Pi).  But it is important to note that this does not imply y is reachable from x. Density reachability is asymmetric.
"))
    density_text_3 <- HTML(paste0("Two points x, y are density connected, if they are density reachable from a point o. Density connectivity is symmetric.
"))
    
    
    density_text_4 <- HTML(paste0("Now, let us see how clusters are formed:
For a given Data Set 'D' and input parameters 'r'and 'n', cluster C is formed if it satisfies two conditions.","<br>", "<b>",
"1. Maximality","</b>","<br>","
Given two observations x, y from the Data Set D, if x belongs to C, and y is Density Reachable from x, then y belongs to C.","<br>",
"<b>","2. Connectivity","</b>","<br>","
For all x,y that belongs to Cluster C, x,y are Density Connected.", "<br>", 
"One major drawback of DBSCAN is that it cannot give best results in the case of varying densities in the data set.", "<br>", 
"Below is an interactive plot for DBSCAN for different input parameters. We can observe that for lower values of 'r', we get more noise
points( in black) and for larger values of 'r', we get bigger clusters with fewer noise points."))
    
    
    output$intro_text9 <- renderText({HTML(paste0("<h1>","DBSCAN","</h1>",
                                                "<p>", "<font face='times new roman', size='4.5'>", density_text_1,"</font>",  "</p>",
                                                "<h3>","Density Reachability","</h3>",
                                                "<p>", "<font face='times new roman', size='4.5'>", density_text_2,"</font>",  "</p>"))})
    
    
    output$intro_text10 <- renderText({HTML(paste0( "<p>", "<font face='times new roman', size='4.5'>",
                                                    "From the above plot we can see that the points p, q are reachable","</font>",  "</p>"
                                                    ,"<h3>","Density Connectivity","</h3>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", density_text_3,"</font>",  "</p>"))})
    
    output$intro_text20 <- renderText({HTML(paste0("<p>", "<font face='times new roman', size='4.5'>","From the above plot we can see that
                                                   the points p, q are density connected by point o","<br>", density_text_4,"</font>",  "</p>"))})
    
    
    output$db1 <- renderImage({
      filename <- normalizePath(file.path(paste('db1', '.png', sep='')))
      list(src = filename,width=400,height=300)
    }, deleteFile = FALSE)
    
    output$db2 <- renderImage({
      filename <- normalizePath(file.path(paste('db2', '.png', sep='')))
      list(src = filename,width=400,height=300)
    }, deleteFile = FALSE)
    
    
    library('dbscan')
    x <- as.matrix(iris[, 1:4])
    clusters_dbscan <- reactive({
      dbscan(x, input$epsilon, input$n)
    })
    
    
   
    
    output$DBSCAN <- renderPlot({
      fviz_cluster(clusters_dbscan(), data = x, main = "DBSCAN", ellipse = TRUE, geom = "point", ggtheme = theme_minimal())
      
    })
    
#____________________________________________________________________________________________________________________________________
    valid_text_1 <- HTML(paste0("All the clustering algorithms we studied depends on some or the other input parameter. For example, K-means algorithm requires number of clusters as an input to determine clusters. 
Performance of any clustering algorithm is highly sensitive to the input parameters and the characteristics of the data set.  Therefore, it is important to choose right parameters to get best fits for the clusters. Cluster validation techniques and indices provide guidelines and insights for choosing these input parameters and determining the goodness of a clustering algorithm."))
    valid_text_2 <- HTML(paste0("Before going to validation techniques, let us have a look at properties of a good cluster. And then we can check how our cluster validation techniques achieve this.
Though there are many valid properties to determine goodness of a clustering algorithm, we can broadly divide them into these three categories."))

    valid_text_3 <- HTML(paste0("1. Compactness","<br>","
This property is implemented by minimizing the intra-cluster variance. Algorithms like k-means, hierarchical, model based clustering fall under this category. These algorithms give better results when the clusters are spherical. They dont give best results for more complicated structures of clusters.","<br>","
2. Connectedness","<br>","
The idea behind this property is that the neighbouring data points should be in same cluster. Density based clustering algorithms fall under this category. They are good at determining arbitrary shaped clusters. But do not give best results when there is more separation between data points.","<br>","
3. Spatial Separation ","<br>","
Determines how far the clusters are from each other. This property when used alone can give trivial solution, hence it is used along with other measures like compactness.
")) 
    
    valid_text_4 <- withMathJax(HTML(paste0("Internal validation techniques are used, when we don't have any prior information about the dataset.
                                This type of validation is completely based on information intrinsic to data set. In general, internal
                                methods examine how compact the cluster is and how seperated it is from the other clusters.", "<br>",
                                "Few of the Internal Validation techniques or indices are:", "<h4>","1. Calinski-Harabasz index","</h4>")))
    
   
    
    valid_text_5 <- HTML(paste0("External validation techniques are used when we know the ground truth about the clusters. This
                                type of validation is based on prior knowledge of the cluster labels.", "<br>", "Few of the External Validation techniques are:"))
    valid_text_6 <- HTML(paste0("Purity provide very little information about the clusters, and may some time lead to trivial solution
                                such as clusters with single element have highest purity. 
                                F-measure considers purity along with completedness to give a holistic overview of the clustering algorithm."))
    
    output$intro_test14 <- renderText({HTML(paste0("<h1>","Cluster Validation","</h1>",
                                                    "<p>", "<font face='times new roman', size='4.5'>", valid_text_1,"</font>",  "</p>",
                                                    "<p>", "<font face='times new roman', size='4.5'>", valid_text_2,"</font>",  "</p>",
                                                    "<p>", "<font face='times new roman', size='4.5'>", valid_text_3,"</font>",  "</p>",
                                                    "<h2>","Types of Cluster Validation","</h2>",
                                                    "<h3>","Internal Validation","</h3>",
                                                    "<p>", "<font face='times new roman', size='4.5'>", valid_text_4,"</font>",  "</p>",
                                                   
                                                   " $$CH = \\frac{trace(S_B)}{trace(S_W)} \\cdot \\frac{n_p - 1}{n_p - k} $$", "<p>", "<font face='times new roman', size='4.5'>",
                                                   "where, ","<br>"," \\(S_W\\) is within cluster scatter matrix","<br>",
                                                   "\\(S_B\\) is between cluster scatter matrix","<br>",
                                                   "n is number of observations","<br>","k is number of clusters","</font>",  "</p>",
                                                    "<h4>","<p>", "<font face='times new roman', size='4.5'>","2. Silhouette Width","</font>","</p>","</h4>", 
                                                   "$$s(i) = \\frac{b(i)-a(i)}{max(b(i), a(i))}  $$",
                                                   "<p>", "<font face='times new roman', size='4.5'>","Silhouette Width value ranges from -1 to 1, \\(a(i)\\)
                                                   determines how compact the cluster is and \\(b(i)\\) determines how seperated the clusters are from each other. Therefore,
                                                   if the SilhouettProjecte Width is closer to 1 the cluster \\(i\\) is more compact and is well seperated from other clusters, which is the preferable case.
                                                   Negative Silhouette Width means that the observations within a cluster are closer to other clusters than 
                                                   the observations in same cluster.","</font>",  "</p>",
                                                    "<h3>","External Validation","</h3>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", valid_text_5,"</font>",  "</p>",
                                                   
                                                   
                                                   "<h4>","1. Purity","</h4>", "<p>", "<font face='times new roman', size='4.5'>",
                                                   "Inorder to calculate purity of a set of clusters, we first have to calculate purity of each cluster ", "\\(P_j\\)", " , which determines the fraction of observations in the cluster that are correctly classified.",
                                                   " Purity of a set of clusters is weighted sum of individual cluster purities.",
                                                   "$$Purity = \\sum_{j=1}^{m} \\frac{n_j}{n}P_j$$","</font>",  "</p>",
                                                   "<h4>","2. F measure","</h4>",
                                                   "<p>", "<font face='times new roman', size='4.5'>", valid_text_6,"</font>",  "</p>",
                                                   "$$F measure = \\frac{Precision \\cdot Recall}{Precision + Recall}$$"))})
    
    
    output$formula7 <- renderUI({
      withMathJax(paste0(""))
    })
 
    conc_text <- HTML(paste0("So far in this tutorial, we discussed several commonly used clustering algorithms. 
    Clustering can be used in variety of applications in image recognition, ratail, websearch etc,. Clustering can also be used as a
                                       preprocessing step for other algorithms.
    Parametric clustering assumes that the clustered data objects belong to some distribution. But non-parametric clustering does not make these assumptions
    Among the non parametric algorithms we discussed distance based and density based algorithms like k-means, hierarchical clustering and DBSCAN. Though there are numerous clustering algorithms
    the type of clustering algorithm to be used is highly dependent on the problem and the data set."))   
    
    output$intro_test19 <- renderText({HTML(paste0("<h1>","Conclusion","</h1>",
                                                  "<p>", "<font face='times new roman', size='4.5'>", conc_text,"</font>",  "</p>",
                                                  "<h1>","References","</h1>", "<br>",
                                                  "1. Arnaud Mignan, 'Mixture modelling from scratch, in R' in towardsdatascience", "<br>",
                                                  "2. George Seif, 'The 5 Clustering Algorithms Data Scientists Need to Know' in towardsdatascience", "<br>",
                                                  "3. Wondong Lee, 'Density-Based Spatial Clustering of Applications with Noise' in fsu lecture notes","<br>",
                                                  "4. Seyoung Kim, 'Clustering: Mixture Models', in CMU lecture notes.", "<br>", 
                                                  "5. https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf", "<br>",
                                                  "6. Bishop text book, ' Pattern Recognition and Machine Learning '", "<br>",
                                                  "7. Julia Handl, Joshua Knowles and Douglas B. Kell , 'Computational cluster validation in post-genomic data
analysis' in ATICS Bioinformatics Advance Access published May 24, 2005", "<br>",
                                                  "8. Erendira Rendon, Itzel, Alenjandra, Arizmendi, 'Internal vs External Validation' in INTERNATIONAL JOURNAL OF COMPUTERS AND COMMUNICATIONS 2011"))})
    
   
})
