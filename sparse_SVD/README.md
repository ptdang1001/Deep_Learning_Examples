# Sparse Matrix SVD

## Description: 

A scheme for calculating the SVD of a matrix. Utilizes the sparse structure for data compression using CSR/CRS/Yale format. 

## Dependencies:

Python + Numpy


## Known bugs:

* "Something weird happened the first time I ran your test code, and it seemed to never complete the 50x50
matrix SVD. It worked on the second run though and hasn't gotten stuck since.. so I'm wondering if there's
maybe a floating point loop invariant somewhere? I'm thinking maybe one side of a condition became NaN
and  the algorithm got stuck. It's also possible that it was something unrelated, but it seems a little weird."



## Resolved bugs:

* On occasion completely skips the iterative step. Probably has to do with the way counting non-zero values is handled, i.e. it thinks it is done when it's really not.

July 2, 2020 : resolved!
