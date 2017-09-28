-   [Example 1 : Reproduce the networks of Sachs et al. (2005)](#example-1-reproduce-the-networks-of-sachs-et-al.-2005)
    -   [exploring the observational data](#exploring-the-observational-data)
    -   [Discretizing the data](#discretizing-the-data)
    -   [Learning the BN from discretized data](#learning-the-bn-from-discretized-data)
    -   [Averaging the network](#averaging-the-network)
    -   [Use interventional data for Sachs](#use-interventional-data-for-sachs)
-   [Example 2: building an epigenetic networks from CLL data](#example-2-building-an-epigenetic-networks-from-cll-data)
    -   [Load pre-compiled epigenetic data matrix for CLL](#load-pre-compiled-epigenetic-data-matrix-for-cll)
    -   [Visualize how these epigenetic features are correlated](#visualize-how-these-epigenetic-features-are-correlated)
    -   [Visualize how these correlated features cluster together](#visualize-how-these-correlated-features-cluster-together)
    -   [Look at the distribution of the data](#look-at-the-distribution-of-the-data)
    -   [Perform discretization using Ckmeans.1d.dp](#perform-discretization-using-ckmeans.1d.dp)
    -   [Learn bayesian network from the discretized data using bnlearn](#learn-bayesian-network-from-the-discretized-data-using-bnlearn)
    -   [Plotting the bayesian network](#plotting-the-bayesian-network)
    -   [Inference : making predictions using the network](#inference-making-predictions-using-the-network)

``` r
library(ggplot2)
library(reshape2)
library(Ckmeans.1d.dp)
library(corrplot)
library(bnlearn)
library(igraph)
library(knitr)
```

Example 1 : Reproduce the networks of Sachs et al. (2005)
=========================================================

This dataset consists of single-cell cytometry data on 11 proteins in a primary T-cell signaling network, using normal condition ("observational") or perturbed conditions ("interventional").

![Literature validated network](validated_net.jpg)

exploring the observational data
--------------------------------

``` r
sachs = read.table(file.path(path,'data','sachs.data.txt'),header=TRUE)

tmp = melt(sachs)
```

    ## No id variables; using all as measure variables

``` r
p = ggplot(tmp,aes(x=value, fill=variable, color=variable)) + 
    geom_density(alpha=0.30,size=0.7) + theme_bw() +
    facet_wrap(~variable, scales = "free") + labs(x="log2(value)")+
    theme(legend.title = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())

rm(tmp)
p
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-2-1.png)

Discretizing the data
---------------------

We discretize the data, since it does not follow the assumption of normal distribution.

``` r
## discretize the data
dsachs = discretize(sachs,method='hartemink',breaks = 3, ibreaks = 60, idisc = 'quantile')

tmp=melt(dsachs,id.vars=c())
```

    ## Warning: attributes are not identical across measure variables; they will
    ## be dropped

``` r
p = ggplot(tmp,aes(x=value, fill=variable, color=variable)) + 
    geom_bar() + theme_bw() +
    facet_wrap(~variable, scales = "free") + labs(x="Categories")+
    theme(legend.title = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()
          )

p
```

![](bnTutorial_files/figure-markdown_github/sachs-discretize-1.png)

Learning the BN from discretized data
-------------------------------------

``` r
boot = boot.strength(data=dsachs, R=500,
                     algorithm='hc',
                     algorithm.args = list(score='bde', iss=10)
)

head(boot)
```

    ##   from   to strength direction
    ## 1  Raf  Mek    1.000 0.5100000
    ## 2  Raf Plcg    0.226 0.5044248
    ## 3  Raf PIP2    0.048 0.4791667
    ## 4  Raf PIP3    0.006 0.5000000
    ## 5  Raf  Erk    0.006 0.3333333
    ## 6  Raf  Akt    0.002 1.0000000

``` r
dim(boot)
```

    ## [1] 110   4

``` r
## now threshold the edge strength and the direction evidence

boot.filt = boot[(boot$strength > 0.85 & boot$direction >= 0.5),]

dim(boot.filt)
```

    ## [1] 10  4

``` r
boot.filt
```

    ##     from   to strength direction
    ## 1    Raf  Mek    1.000 0.5100000
    ## 23  Plcg PIP2    1.000 0.5150000
    ## 24  Plcg PIP3    1.000 0.5220000
    ## 34  PIP2 PIP3    1.000 0.5070000
    ## 56   Erk  Akt    1.000 0.5740000
    ## 57   Erk  PKA    0.994 0.5845070
    ## 67   Akt  PKA    1.000 0.5820000
    ## 89   PKC  P38    1.000 0.5060000
    ## 90   PKC  Jnk    1.000 0.5080000
    ## 100  P38  Jnk    0.918 0.5043573

Averaging the network
---------------------

We have learned 500 networks starting from different initial random networks; we can now average these networks.

``` r
avg.boot = averaged.network(boot, threshold = 0.85)
net = avg.boot$arcs

net = graph_from_edgelist(net,directed=T)
library(Rgraphviz)
```

    ## Loading required package: graph

    ## Loading required package: BiocGenerics

    ## Loading required package: parallel

    ## 
    ## Attaching package: 'BiocGenerics'

    ## The following objects are masked from 'package:parallel':
    ## 
    ##     clusterApply, clusterApplyLB, clusterCall, clusterEvalQ,
    ##     clusterExport, clusterMap, parApply, parCapply, parLapply,
    ##     parLapplyLB, parRapply, parSapply, parSapplyLB

    ## The following objects are masked from 'package:igraph':
    ## 
    ##     normalize, union

    ## The following object is masked from 'package:bnlearn':
    ## 
    ##     score

    ## The following objects are masked from 'package:stats':
    ## 
    ##     IQR, mad, xtabs

    ## The following objects are masked from 'package:base':
    ## 
    ##     anyDuplicated, append, as.data.frame, cbind, colnames,
    ##     do.call, duplicated, eval, evalq, Filter, Find, get, grep,
    ##     grepl, intersect, is.unsorted, lapply, lengths, Map, mapply,
    ##     match, mget, order, paste, pmax, pmax.int, pmin, pmin.int,
    ##     Position, rank, rbind, Reduce, rownames, sapply, setdiff,
    ##     sort, table, tapply, union, unique, unsplit, which, which.max,
    ##     which.min

    ## 
    ## Attaching package: 'graph'

    ## The following objects are masked from 'package:igraph':
    ## 
    ##     degree, edges, intersection

    ## The following objects are masked from 'package:bnlearn':
    ## 
    ##     degree, nodes, nodes<-

    ## Loading required package: grid

``` r
graphviz.plot(avg.boot)
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-4-1.png)

Use interventional data for Sachs
---------------------------------

So far the network was built using observational data, i.e. data collected in normal conditions. The dataset however contained additional interventional data, resulting from perturbation of specific nodes in the network. This data can either be used for validation (do the perturbation have the predicted effect from the initial network ?) or can be used to strengthen the evidence on the learned edges.

``` r
isachs = read.table(file.path(path,'data',"sachs.interventional.txt"), header = TRUE, colClasses = "factor")


head(isachs)
```

    ##   Raf Mek Plcg PIP2 PIP3 Erk Akt PKA PKC P38 Jnk INT
    ## 1   1   1    1    2    3   2   1   3   1   2   1   8
    ## 2   1   1    1    1    3   3   2   3   1   2   1   8
    ## 3   1   1    2    2    3   2   1   3   2   1   1   8
    ## 4   1   1    1    1    3   2   1   3   1   3   1   8
    ## 5   1   1    1    1    3   2   1   3   1   1   1   8
    ## 6   1   1    1    1    2   2   1   3   1   2   1   8

``` r
INT = sapply(1:11, function(x) { which(isachs$INT == x) })

isachs = isachs[, 1:11]
nodes = names(isachs)
names(INT) = nodes

## we start from a set of 200 random graphs
start = random.graph(nodes = nodes,  method = "melancon", num = 200, burn.in = 10^5, every = 100)

netlist = lapply(start, function(net) {
  tabu(isachs, score = "mbde", exp = INT, iss = 1, start = net, tabu = 50) }
  )

arcs = custom.strength(netlist, nodes = nodes, cpdag = FALSE)

bn.mbde = averaged.network(arcs, threshold = 0.85)
```

Example 2: building an epigenetic networks from CLL data
========================================================

Here, we want to build a chromatin network where nodes are epigenetic components, and the observations are the state of these epigenetic modifications at the promoters of genes.

Load pre-compiled epigenetic data matrix for CLL
------------------------------------------------

``` r
dat = read.table(paste0(path,"data/CEMT_26-nonCGI-allDataContinuous.txt"), 
                 header = TRUE, stringsAsFactors = FALSE, sep="\t")

head(dat)
```

    ##           CPGfrac RPKM H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me3 H3K9me3
    ## 10            7.5    0       7       19        8       7       6      14
    ## 10000         9.0    5      13        7       11      11       3      15
    ## 10001         0.0   19      10        8       11      28      77       5
    ## 10003         0.0    1      18       40       10      22       9      17
    ## 100033413    10.0    0      12       11       37       7       6      44
    ## 100033414    10.0    0      13       11       37       9       9      46
    ##           Input
    ## 10            9
    ## 10000         8
    ## 10001         3
    ## 10003        12
    ## 100033413    20
    ## 100033414    19

``` r
dim(dat)
```

    ## [1] 5964    9

``` r
dat[,3:ncol(dat)] = round(dat[,3:ncol(dat)]/dat$Input,3)
dat = dat[,-ncol(dat)]

corVals = round(cor(dat, method="spearman"),2)
```

|          |  CPGfrac|   RPKM|  H3K27ac|  H3K27me3|  H3K36me3|  H3K4me1|  H3K4me3|  H3K9me3|
|----------|--------:|------:|--------:|---------:|---------:|--------:|--------:|--------:|
| CPGfrac  |     1.00|  -0.06|    -0.36|     -0.17|      0.14|    -0.38|    -0.42|    -0.05|
| RPKM     |    -0.06|   1.00|     0.24|     -0.48|      0.03|     0.28|     0.20|    -0.36|
| H3K27ac  |    -0.36|   0.24|     1.00|      0.24|      0.43|     0.72|     0.72|     0.30|
| H3K27me3 |    -0.17|  -0.48|     0.24|      1.00|      0.37|     0.17|     0.28|     0.73|
| H3K36me3 |     0.14|   0.03|     0.43|      0.37|      1.00|     0.34|     0.40|     0.57|
| H3K4me1  |    -0.38|   0.28|     0.72|      0.17|      0.34|     1.00|     0.59|     0.16|
| H3K4me3  |    -0.42|   0.20|     0.72|      0.28|      0.40|     0.59|     1.00|     0.38|
| H3K9me3  |    -0.05|  -0.36|     0.30|      0.73|      0.57|     0.16|     0.38|     1.00|

Visualize how these epigenetic features are correlated
------------------------------------------------------

``` r
corrplot.mixed(corVals, tl.col = "black", tl.cex=0.8)
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-8-1.png)

Visualize how these correlated features cluster together
--------------------------------------------------------

``` r
corrplot(corVals, order = "hclust", hclust.method = "ward.D2", addrect = 2, tl.col = "black");rm(corVals)
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-9-1.png)

Look at the distribution of the data
------------------------------------

``` r
summary(dat)
```

    ##     CPGfrac            RPKM             H3K27ac          H3K27me3     
    ##  Min.   : 0.000   Min.   :   0.000   Min.   : 0.111   Min.   : 0.050  
    ##  1st Qu.: 5.000   1st Qu.:   0.000   1st Qu.: 0.762   1st Qu.: 0.643  
    ##  Median : 8.000   Median :   0.000   Median : 1.000   Median : 1.176  
    ##  Mean   : 6.829   Mean   :   6.139   Mean   : 1.783   Mean   : 2.071  
    ##  3rd Qu.: 9.000   3rd Qu.:   1.000   3rd Qu.: 1.750   3rd Qu.: 1.857  
    ##  Max.   :10.000   Max.   :6557.000   Max.   :63.000   Max.   :42.000  
    ##     H3K36me3         H3K4me1          H3K4me3           H3K9me3      
    ##  Min.   : 0.150   Min.   : 0.077   Min.   :  0.083   Min.   : 0.053  
    ##  1st Qu.: 0.611   1st Qu.: 0.500   1st Qu.:  0.368   1st Qu.: 0.467  
    ##  Median : 0.857   Median : 0.857   Median :  0.500   Median : 0.722  
    ##  Mean   : 1.693   Mean   : 1.980   Mean   :  1.521   Mean   : 1.424  
    ##  3rd Qu.: 1.556   3rd Qu.: 2.113   3rd Qu.:  1.093   3rd Qu.: 1.200  
    ##  Max.   :43.000   Max.   :80.000   Max.   :104.000   Max.   :52.000

``` r
log2dat = log2(dat+0.1)
tmp = melt(log2dat)
```

    ## No id variables; using all as measure variables

``` r
p = ggplot(tmp,aes(x=value, fill=variable, color=variable)) + 
    geom_density(alpha=0.30,size=0.7) + theme_bw() +
    facet_wrap(~variable, scales = "free") + labs(x="log2(value)")+
    theme(legend.title = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
rm(tmp)
p
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-10-1.png)

Perform discretization using Ckmeans.1d.dp
------------------------------------------

For details regarding Ckmeans.1d.dp, see <https://cran.r-project.org/web/packages/Ckmeans.1d.dp/vignettes/Ckmeans.1d.dp.html> Lets check the discrete states for H3K4me3 as an example

``` r
res = Ckmeans.1d.dp(log2dat$H3K4me3)
# Number of clusters predicted
max(res$cluster)
```

    ## [1] 3

``` r
# Lets look at the predictions
plot(log2dat$H3K4me3, col= res$cluster, cex=0.5, pch=20, xlab="Genes", ylab= "H3K4me3 signal")
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-11-1.png)

``` r
#abline(h=res$centers, lwd=2,lty=2,col="blue")
rm(res)

# Compute optimal states for all
states = apply(log2dat, 2, function(y){max(Ckmeans.1d.dp(x=y, k=c(1,9))$cluster)})
```

    ## Warning in cluster.1d.dp(x, k, y, method, estimate.k, "L2", deparse(substitute(x)), : Max number of clusters used. Consider increasing k!

``` r
# For methylation lots of states are predicted, from the distribution plots
# its clear that these are mostly intermediate states
# so lets approximate by taking 3 states

res = Ckmeans.1d.dp(log2dat$CPGfrac, k=c(1,3))
```

    ## Warning in cluster.1d.dp(x, k, y, method, estimate.k, "L2", deparse(substitute(x)), : Max number of clusters used. Consider increasing k!

``` r
plot(log2dat$CPGfrac, col= res$cluster, cex=0.5, pch=20, xlab="Genes", ylab= "CPGfrac")
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-11-2.png)

``` r
#abline(h=res$centers, lwd=2,lty=2,col="blue")

# We accept all the predicted states from Ckmeans.1d.dp except for methylation
# Final max discrete states are
states[1] = 3
states
```

    ##  CPGfrac     RPKM  H3K27ac H3K27me3 H3K36me3  H3K4me1  H3K4me3  H3K9me3 
    ##        3        6        2        3        3        3        3        2

``` r
# Creating the discretized version of the data
disDat = matrix(ncol=ncol(log2dat), nrow=nrow(log2dat))
colnames(disDat) = colnames(log2dat)
rownames(disDat) = rownames(log2dat)
disDat = as.data.frame(disDat)

for( i in 1: length(states))
{
  tmp = suppressWarnings(Ckmeans.1d.dp(log2dat[,i], k=c(1,states[i])))
  disDat[,i] = factor(tmp$cluster)
  rm(tmp)
}
rm(i, states)

# Final view of the discretized data
summary(disDat)
```

    ##  CPGfrac  RPKM     H3K27ac  H3K27me3 H3K36me3 H3K4me1  H3K4me3  H3K9me3 
    ##  1: 647   1:4382   1:4562   1:2133   1:3629   1:3396   1:3819   1:4890  
    ##  2: 519   2: 354   2:1402   2:3178   2:1743   2:1900   2:1189   2:1074  
    ##  3:4798   3: 394            3: 653   3: 592   3: 668   3: 956           
    ##           4: 420                                                        
    ##           5: 313                                                        
    ##           6: 101

``` r
# Plot the discretized data
tmp=melt(disDat,id.vars=c())
```

    ## Warning: attributes are not identical across measure variables; they will
    ## be dropped

``` r
p = ggplot(tmp,aes(x=value, fill=variable, color=variable)) + 
    geom_bar() + theme_bw() +
    facet_wrap(~variable, scales = "free") + labs(x="Categories")+
    theme(legend.title = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()
          )

p
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-11-3.png)

Learn bayesian network from the discretized data using bnlearn
--------------------------------------------------------------

We use a set of blacklist edges: no edge should go out of the expression node

``` r
# Black list .. no interaction should originate from gene expression
bl=data.frame(from="RPKM",to=colnames(disDat))
bl=bl[-which(bl$to == "RPKM"),]

# Learn bayesian network via bootstrapping
strength = boot.strength(disDat, algorithm="tabu", 
                         algorithm.args=list(score="aic", tabu=10, blacklist=bl), 
                         R=1000) # takes about 2mins
rm(bl)

# Select only those interactions meeting our filteration criteria
selarcs = strength[strength$direction > 0.5 ,]
selarcs = selarcs[order(selarcs$strength,decreasing=TRUE),]
selarcs
```

    ##        from       to strength direction
    ## 5   CPGfrac  H3K4me1    1.000 0.6205000
    ## 6   CPGfrac  H3K4me3    1.000 0.5005000
    ## 17  H3K27ac H3K27me3    1.000 0.5080000
    ## 19  H3K27ac  H3K4me1    1.000 0.5020000
    ## 23 H3K27me3     RPKM    1.000 1.0000000
    ## 25 H3K27me3 H3K36me3    1.000 0.5040000
    ## 28 H3K27me3  H3K9me3    1.000 0.5270000
    ## 47  H3K4me3 H3K36me3    1.000 0.5005000
    ## 54  H3K9me3 H3K36me3    1.000 0.5155000
    ## 56  H3K9me3  H3K4me3    1.000 0.5005000
    ## 3   CPGfrac H3K27me3    0.999 0.5055055
    ## 30 H3K36me3     RPKM    0.995 1.0000000
    ## 44  H3K4me3     RPKM    0.989 1.0000000
    ## 18  H3K27ac H3K36me3    0.954 0.5131027
    ## 33 H3K36me3  H3K4me1    0.948 0.6165612
    ## 2   CPGfrac  H3K27ac    0.805 0.5012422
    ## 21  H3K27ac  H3K9me3    0.732 0.5150273
    ## 29 H3K36me3  CPGfrac    0.494 0.5050607
    ## 7   CPGfrac  H3K9me3    0.319 0.5015674
    ## 16  H3K27ac     RPKM    0.248 1.0000000
    ## 42  H3K4me1  H3K9me3    0.188 0.6063830
    ## 37  H3K4me1     RPKM    0.017 1.0000000
    ## 1   CPGfrac     RPKM    0.006 1.0000000

``` r
# Get an averaged network
dag.average.filt = averaged.network(selarcs, threshold = 0.90)
```

``` r
kable(dag.average.filt$arcs, caption = "Interactions meeting our threshold for strength and direction")
```

| from     | to       |
|:---------|:---------|
| CPGfrac  | H3K27me3 |
| CPGfrac  | H3K4me3  |
| CPGfrac  | H3K4me1  |
| H3K27ac  | H3K27me3 |
| H3K27ac  | H3K36me3 |
| H3K27ac  | H3K4me1  |
| H3K27me3 | H3K9me3  |
| H3K27me3 | H3K36me3 |
| H3K27me3 | RPKM     |
| H3K4me3  | H3K36me3 |
| H3K4me3  | RPKM     |
| H3K9me3  | H3K4me3  |
| H3K9me3  | H3K36me3 |
| H3K36me3 | H3K4me1  |
| H3K36me3 | RPKM     |

``` r
# Compute correlations for the selected interactions
net = dag.average.filt$arcs
corVals = apply(net,1, function(x){
                                    a = log2dat[,x[1]]
                                    b = log2dat[,x[2]]
                                    c = round(cor(a,b,method="spearman"),2)
                                    return(c)
                                  })

# Coloring the correlation values
col = rep("forestgreen",nrow(net))
col[which(corVals < 0)] = "firebrick"

# Setting up the network in igraph
net = graph_from_edgelist(net,directed=T)
E(net)$weight = corVals
V(net)$label.color="black"
rm(corVals)
```

Plotting the bayesian network
-----------------------------

``` r
plot(net, edge.color=col, vertex.shape="circle", vertex.size = 40,layout=l, edge.width=E(net)$weight*3.5+3)
box();rm(col,l,pos.grp,neg.grp,left.grp,right.grp,mid.grp)
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-16-1.png)

Inference : making predictions using the network
------------------------------------------------

``` r
# Fitting the learnt network to the data to obtain
# conditional probability tables
fitted = bn.fit(dag.average.filt, disDat)


# Predict Expression level by setting DNAme to high/low
x.high = as.numeric(table(cpdist(fitted,nodes=c('RPKM'),CPGfrac=='3')))
x.low = as.numeric(table(cpdist(fitted,nodes=c('RPKM'),CPGfrac=='1')))

pred = data.frame(prop=c(x.high/sum(x.high),x.low/sum(x.low)),
                  condition=c(rep('DNAme high',length(x.high)),rep('DNAme low',length(x.high))),
                  Expression=c(1:length(x.high),1:length(x.low))
)

ggplot(pred,aes(x=Expression,fill=condition,y=prop)) + geom_bar(stat='identity',position=position_dodge())
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-17-1.png)

``` r
# Predict Expression level by setting H3K4me1 to high/low
x.high = as.numeric(table(cpdist(fitted,nodes=c('RPKM'),H3K4me1=='3')))
x.low = as.numeric(table(cpdist(fitted,nodes=c('RPKM'),H3K4me1=='1')))

pred = data.frame(prop=c(x.high/sum(x.high),x.low/sum(x.low)),
                  condition=c(rep('K4me1 high',length(x.high)),rep('K4me1 low',length(x.high))),
                  Expression=c(1:length(x.high),1:length(x.low))
)

ggplot(pred,aes(x=Expression,fill=condition,y=prop)) + geom_bar(stat='identity',position=position_dodge())
```

![](bnTutorial_files/figure-markdown_github/unnamed-chunk-18-1.png)
