import cc.mallet.util.*;
import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.topics.*;

import java.util.*;
import java.util.regex.*;
import java.io.*;

import java.util.Iterator;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

class ArrayIndexComparator implements Comparator<Integer>
{
    private final Double[] array;

    public ArrayIndexComparator(Double[] array)
    {
        this.array = array;
    }

    public Integer[] createIndexArray()
    {
        Integer[] indexes = new Integer[array.length];
        for (int i = 0; i < array.length; i++)
            indexes[i] = i;
        
        return indexes;
    }

    @Override
    public int compare(Integer index1, Integer index2)
    {
        return array[index2].compareTo(array[index1]);
    }
}

/**
 * container for word_idx proba pairs
 */
class WordProbaPair {
	private int word_idx;
	private double word_proba;
	
	public WordProbaPair(double word_proba){
		this.word_idx = -1;
		this.word_proba = word_proba;
	}
	
	public WordProbaPair(){
		this.word_idx = -1;
		this.word_proba = 0;
	}
	
	public WordProbaPair(int word_idx, double word_proba){
		this.word_idx = word_idx;
		this.word_proba = word_proba;
	}
	
	public int get_word_idx(){
		return this.word_idx;
	}
	
	public double get_word_proba(){
		return this.word_proba;
	}
	
	public void set_word_proba(double word_proba){
		this.word_proba = word_proba;
	}
}

public class MalletLDAModel {

   /* Module using the mallet LDA implementation based on gibbs sampling.
    * Gensim uses variational bayes for the inference. Gibbs sampling might be
    * better to perform the inference on small documents.
    */
	
	/**
	 * take a trained model as input and return the phi = P(word |topic), the words are sorted by proba
	 * @param model: an instance of ParallelTopicModel
	 * @param num_topics 
	 * @param vocab_size: the size of the vocabulary.
	 * @return an ArrayList of phi for each topic.
	 */
	public static ArrayList<ArrayList<WordProbaPair>> compute_phi(ParallelTopicModel model, int num_topics, int vocab_size){
		ArrayList<ArrayList<WordProbaPair>> phi = new ArrayList<ArrayList<WordProbaPair>>();
		ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();
		for (int topic = 0; topic < num_topics; topic++) {
            Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();
            ArrayList<WordProbaPair> current_phi = new ArrayList<WordProbaPair>(Arrays.asList(new WordProbaPair[vocab_size]));
            int word_position = 0;
            double sum_current_phi = 0;
            while(iterator.hasNext()){	//we fill all phi with beta + occurrences, we will need to normalize afterwards to get a distribution
            	IDSorter idCountPair = iterator.next();
            	int word_idx = idCountPair.getID();
            	double occurrences = (double) idCountPair.getWeight();
            	current_phi.set(word_position, new WordProbaPair(word_idx,model.beta + occurrences));
            	sum_current_phi += occurrences;
            	word_position += 1;
            }
            sum_current_phi += vocab_size*model.beta;
            for(int idx=0; idx<current_phi.size();++idx){	//here we normalize
            	if(current_phi.get(idx) == null){
            		current_phi.set(idx, new WordProbaPair(model.beta/sum_current_phi)); //if the word is not referenced then its probability is beta/sum_current_phi
            	} else {
            		current_phi.get(idx).set_word_proba(current_phi.get(idx).get_word_proba()/sum_current_phi);
            	}
            }
            phi.add(current_phi);
		}
		return phi;
	}
	
	/**
	 * for a given phi and theta distribution, return the most probable words/probabilities.
	 * @param phi
	 * @param theta
	 * @param top_k: the number of best word/proba pairs we want.
	 * @return
	 */
	public static ArrayList<WordProbaPair> compute_top_words(ArrayList<ArrayList<WordProbaPair>> phi, double[] theta, int top_k){
		ArrayList<WordProbaPair> top_wordidx_proba = new ArrayList<WordProbaPair>(top_k);
		Double[] scores = new Double[phi.get(0).size()];
		System.out.println(Arrays.toString(theta));
		for(int topic=0; topic<phi.size(); ++topic){
			double theta_value = theta[topic];
			for(WordProbaPair w_p : phi.get(topic)){
				int word_idx = w_p.get_word_idx();
				double proba = theta_value*w_p.get_word_proba();
				if(word_idx == -1){
					continue;
				} else if(scores[word_idx] == null){
					scores[word_idx] = 0.0;
				}
				scores[word_idx] += proba;
			}
		}
		ArrayIndexComparator comparator = new ArrayIndexComparator(scores);
		Integer[] sorted_indexes = comparator.createIndexArray();
		Arrays.sort(sorted_indexes,comparator);
		for(int idx=0;idx<top_k;++idx){
			top_wordidx_proba.add(new WordProbaPair(sorted_indexes[idx],scores[sorted_indexes[idx]]));
		}
		return top_wordidx_proba;
	}

    public static void main(String []args) {
    	
    	String DATA_PATH = "/home/mat/Documents/Git/GroovesharkRecommender/Recommender/Data_processing/tokensSeqs_small_corpus.json";
    	int NUM_TOPICS = 150;
    	int NUM_ITERATIONS = 1200;
    	ArrayList<String> CORPUS = new ArrayList<String>();
    	String NEW_DOCUMENT = "13 13 13 13 13 4137 4137 4137 9836 9836 9836 2031 2031 2031 2031 2031 409 409 409 409 409 382";
    	
    	/*
    	 * Parsing the json file
    	 */
    	JSONParser parser = new JSONParser();
    	HashMap corpus = new HashMap();
    	HashMap invLexicon = new HashMap();
    	
    	try {
    		Object obj = parser.parse(new FileReader(DATA_PATH));
    		JSONObject jsonObject = (JSONObject) obj;
    		
    		corpus = (HashMap) jsonObject.get("corpus");
    		invLexicon = (HashMap) jsonObject.get("invLexicon");
    	} catch (FileNotFoundException e) {
    		e.printStackTrace();
    	} catch (IOException e) {
    		e.printStackTrace();
    	} catch (ParseException e) {
    		e.printStackTrace();
    	}
    	System.out.println("corpus size:"+corpus.size());
    	System.out.println("lexicon size:"+invLexicon.size());
    	/*
    	 * Formatting the data to a list of string. Each string is one playlist and the tokens are 
    	 * separated by on space. 
    	 */
    	Iterator<String> iter = corpus.keySet().iterator();
    	while(iter.hasNext()){
    		String playlist_id = (String) iter.next();
    		ArrayList<String> playlist = (ArrayList<String>) corpus.get(playlist_id); 
    		String token_string = "";
    		for(int idx=0; idx<playlist.size();++idx){
    			token_string += ' '+(String) playlist.get(idx);
    		}
    		CORPUS.add(token_string);
    	}
    	System.out.println(CORPUS.get(0));
    	
    	// Begin by importing documents from text to feature sequences
        ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
        Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");
        pipeList.add(new CharSequence2TokenSequence(tokenPattern));
        pipeList.add(new TokenSequenceLowercase());
        pipeList.add(new TokenSequence2FeatureSequence());
        //pipeList.add(new StringList2FeatureSequence());
        
        InstanceList instances = new InstanceList(new SerialPipes(pipeList));
        System.out.println("Creating instances");
        for(int playlist_idx=0; playlist_idx<CORPUS.size(); ++playlist_idx){
        	instances.addThruPipe((new Instance(CORPUS.get(playlist_idx),Integer.toString(playlist_idx) ,"Instance-"+Integer.toString(playlist_idx),null)));
        	//System.out.println(instances.get(playlist_idx).getAlphabet());
        }
    	
        // Create a model with 100 topics, alpha_t = 0.005, beta_w = 0.5
        //  Note that the first parameter is passed as the sum over topics, while
        //  the second is the parameter for a single dimension of the Dirichlet prior.
        
        ParallelTopicModel model = new ParallelTopicModel(NUM_TOPICS, 0.5, 0.5);

        model.addInstances( instances);
        model.setNumThreads(3);
        model.setNumIterations(NUM_ITERATIONS);
        try{
        	model.estimate();
        } catch(Exception e) {
        	e.printStackTrace();
        }

        // The data alphabet maps word IDs to strings
        Alphabet dataAlphabet = instances.getDataAlphabet();
        
        FeatureSequence tokens = (FeatureSequence) model.getData().get(0).instance.getData();
        LabelSequence topics = model.getData().get(0).topicSequence;
        
        Formatter out = new Formatter(new StringBuilder(), Locale.US);
        for (int position = 0; position < tokens.getLength(); position++) {
            out.format("%s-%d ", dataAlphabet.lookupObject(tokens.getIndexAtPosition(position)), topics.getIndexAtPosition(position));
        }
        System.out.println(out);
        
        // Estimate the topic distribution of the first instance, 
        //  given the current Gibbs state.
        double[] topicDistribution = model.getTopicProbabilities(0);

        // Get an array of sorted sets of word ID/count pairs
        ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();
        
        // Show top 5 words in topics with proportions for the first document
        for (int topic = 0; topic < NUM_TOPICS; topic++) {
            Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();
            
            out = new Formatter(new StringBuilder(), Locale.US);
            out.format("%d\t%.3f\t", topic, topicDistribution[topic]);
            int rank = 0;
            while (iterator.hasNext() && rank < 10) {
                IDSorter idCountPair = iterator.next();
                out.format("%s (%.0f)  ", invLexicon.get(dataAlphabet.lookupObject(idCountPair.getID())), idCountPair.getWeight());
                rank++;
            }
            System.out.println(out);
        }
        
        //retrieve the phi
        ArrayList<ArrayList<WordProbaPair>> phi = compute_phi(model,NUM_TOPICS,invLexicon.size());
        		
    	//printing the new document:
    	String[] tokens_list = NEW_DOCUMENT.split(" ");
    	System.out.println("Incomming document: ");
    	for(int idx=0; idx< tokens_list.length; ++idx){
    		System.out.print(invLexicon.get(tokens_list[idx])+"  ");
    	}
    	System.out.println("");

        // Create a new instance named "test instance" with empty target and source fields for the new document.
        InstanceList testing = new InstanceList(instances.getPipe());
        testing.addThruPipe(new Instance(NEW_DOCUMENT, null, "test instance", null));

        TopicInferencer inferencer = model.getInferencer();
        double[] theta = inferencer.getSampledDistribution(testing.get(0), 300, 1, 5);
        
        //print the top words for the new document
        ArrayList<WordProbaPair> top_wordidx_proba = new ArrayList<WordProbaPair>();
        top_wordidx_proba = compute_top_words(phi,theta,30);
        for(WordProbaPair word_proba: top_wordidx_proba){
        	if(!Arrays.asList(tokens_list).contains(dataAlphabet.lookupObject(word_proba.get_word_idx()))){
        		String artist = (String) invLexicon.get(dataAlphabet.lookupObject(word_proba.get_word_idx()));
            	double proba = word_proba.get_word_proba();
            	System.out.println(artist+"  "+proba);
        	}
        }
        
    }
}
