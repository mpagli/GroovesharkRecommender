### Matrix Factorization for collaborative filtering

#### Basic idea

In this experiment we tried to use truncated SVD factorization to create a model of our playlists of artists. Lets consider we have 9 artists, and three playlists stored in a matrix R:

        | _ 1 _ 3 _ 3 _ _ _ | <- playlist1 has 1 song from artist 2, 3 from artist 4 ... 
    R = | _ _ _ 4 2 _ 1 _ 1 |
        | 5 3 1 _ _ _ _ 3 _ |

The assumption is that we know only few values of R. This assumption is not totally justified in our case since a playlist represent the deliberate action of a user to associate several songs together. In opposition to a set of movie ratings where the movies with no rating carry no information. We will come back to that point later on. 

If we can approximate R as the product of two matrix P and Q of sizes respectively 3xk and kx9:

    R ~ P.Q

Then the rows of P can be seen as a decomposition of the first playlist into k features, the columns of Q as the decomposition into k features of one artist. The rows of Q are a basis of dimension k for our artists, and constitute our  model. I a new user/playlist arrive and we know that user like feature 1 a lot, completely dislike feature 2: p = [3,-1]. Then we can compute an appreciation score for each item/artist by multiplying p by the item model Q.   

To find these matrices P and Q one solution is to perform gradient descent on the known values. In the case there are no missing values we can perform a classic SVD decomposition.

When a new user arrive with only few known ratings, we can estimate his feature decomposition p doing the same gradient descent procedure as for the training but with Q fixed. p represent the user projected in the item feature space, we get all the missing values of that user computing p.Q.

#### Our pipeline

We are using the [pySVD](http://code.google.com/p/pyrsvd/) module. This module handles the gradient descent procedure and save the two matrices P and Q. Only Q is interesting for us. When a new playlist comes in we use MF_recommender.py to get the decomposition into feature using gradient descent and generate a recommendation.

Before computing the model we are doing some preprocessing:

* all the artist with less than 300 counts in the database are discarded
* all the playlists with less than 15 artists are discarded.

This processing is quite restricting and we end up only with 49415 playlists on 6911 artists to perform the training, validation and testing.    


