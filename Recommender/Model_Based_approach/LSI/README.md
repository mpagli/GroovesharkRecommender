### Latent Semantic Indexing

#### Basic idea

The idea is exactly the same as for the matrix factorization (MF) method except that the assumption on the data is different. With MF we consider that a lot of values are unknown to us, here we have all the data. When a user creates a playlist, the artists not included are as important as the artists included. When we tried to generate a PQ approximation of our data using gradient descent on the non-zeros values only, we were in fact omitting a lot of essential discriminative information. Trying to learn a model of our playlist considering only artists that are supposed to go well together is bound to fail, since we are only giving positive correlation measures and will end up with a system stating that everything correlates with everything. The really valuable information contained in the playlists is what is inside the playlist and what isn't. Latent Semantic Indexing consist in a truncated SVD with no unknown values. It is mainly used in natural language processing, as a topic modeling method.

Since we know all the values of the matrix we are trying to approximate we can use any standard algorithm to compute the SVD decomposition. Our dataset being too big for the standard np.linalg.svd method, we used (gensim)[http://radimrehurek.com/gensim/index.html]. 

When a knew document d comes in, we can get his distribution of topics by projecting it on the item matrix Q:

		projection = Q.d

In our implementation the document representation selected is tfIdf.

Once we have the projection in the topic space we can get an estimated reconstruction by doing:

		reconstruction = Q^T.projection = Q^T.Q.d

Our recommendation consist in the artist with highest associated score in the reconstruction.

#### Results

For the following input playlist: 

		artist name              count
		Lynyrd Skynyrd :           1
		Black Sabbath :            5
		Metallica :                5
		Iron Maiden :              3
		Jimi Hendrix :             5
		John Zorn :                3

We used LSI with 50 topics, here are the top artists recommended:

		AC/DC :                         1.72452021092
		Led Zeppelin :                  0.801362748479
		Guns N' Roses :                 0.75295889035
		Nirvana :                       0.606058197132
		Creedence Clearwater Revival :  0.51452620951
		Avenged Sevenfold :             0.433649939444
		The Who :                       0.407605261799
		The Doors :                     0.402655746577
		Slipknot :                      0.391221350185
		Pantera :                       0.383772682469
		Marilyn Manson :                0.362612115394
		KISS :                          0.348078329549
		Van Halen :                     0.342707486781
		Red Hot Chili Peppers :         0.330784144559
		Soundgarden :                   0.330335441802
		Tool :                          0.294092692053
		The Rolling Stones :            0.278719358102
		Aerosmith :                     0.277122350057
		Motörhead :                     0.275817028339
		Rob Zombie :                    0.270443358869
		Pearl Jam :                     0.259448272294
		Alice in Chains :               0.245551205416
		Limp Bizkit :                   0.219811782515
		Papa Roach :                    0.216097877216
		Rush :                          0.211276890424
		Bob Marley :                    0.2069613264
		Godsmack :                      0.205724591336
		Megadeth :                      0.203392813396
		Dropkick Murphys :              0.199644939015
		ZZ Top :                        0.19939238713
		Ozzy Osbourne :                 0.197460731894
		Rammstein :                     0.193414822527
		Nine Inch Nails :               0.187452402971
		Queens of the Stone Age :       0.186089291595
		System of a Down :              0.185134743087
		Scorpions :                     0.182191513069
		Rage Against the Machine :      0.181633206379
		Eric Clapton :                  0.176062457475
		Cream :                         0.17350854083
		Foo Fighters :                  0.172442565115

This recommendation seems coherent to me. We would need a more formal evaluation on a test set to really measure how good this system is. 

#### Limitations

LSI is a method to model documents as a linear combination of topics. Creating a recommendation system based on this will successfully find the "big clusters" inside our data. However those big clusters are primarily directed by very popular items and our recommendation engine is lacking of serendipity. One solution could be to rank the playlists according to a sparsity score: playlists with a sparse features representation (only few well defined topics) might be given a lower weight than playlists containing many topics. That way we would promote playlists linking to a lot of diverse topics.

     

