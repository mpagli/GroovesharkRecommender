###About the data format

Two ways have been considered to handle the data:

* Use MongoDB: this was the first attempt, the neighbor-based module uses this approach. Since the unprocessed data consists in a lot of JSON files MongoDB seemed appropriate. However we don't really want to query the dataset, we only want to stream it through our pipeline. 

* Preprocess the data and save everything in a compact JSON file: removing the unnecessary attributes, avoiding redundancies as well as removing playlists that are too short to contain relevant information allows to generate a new dataset. This dataset (around 600Mb) can be loaded in main memory. This dataset is used by the model-based approach.  
