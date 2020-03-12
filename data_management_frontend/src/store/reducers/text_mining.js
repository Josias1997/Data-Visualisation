import * as actionTypes from "./../actions/actionTypes";
import { updateObject } from "../../utility/utility";

const initialState = {
	loading: false,
	error: false,
	predicted: false,
	retweets: '',
	languages_used: '',
	original_authors_retweets: '',
	tweets_by_month: '',
	correlation_matrix: '',
	popular_hashtags_used: '',
	hsh_wrds_trump: '',
	hsh_wrds_hillary: '',
	popular_twitter_account_references: '',
	acc_wrds_trump: '',
	acc_wrds_hillary: '',
	pop_wrds_trump: '',
	pop_wrds_hillary: '',
	popular_negative_words: '',
	pw_trump: '',
	pw_hillary: '',
	nw_trump: '',
	nw_hillary: '',
	sentiment_of_tweets: '',
	average_retweets: '',
	classifier_trump: '',
	classifier_hillary: '',
	tweets_trump: '',
	tweets_hillary: '',
	log_reg_trump: '',
	log_reg_hillary: '',
	svm_trump: '',
	svm_hillary: '',

}

const processingStart = (state) => {
	return updateObject(state, {
		loading: true,
	});
};

const predictSuccess = (state, action) => {
	return updateObject(state, {
		retweets: action.data.retweets,
		languages_used: action.data.languages_used,
		original_authors_retweets: action.data.original_authors_retweets,
		tweets_by_month: action.data.tweets_by_month,
		correlation_matrix: action.data.correlation_matrix,
		popular_hashtags_used: action.data.popular_hashtags_used,
		hsh_wrds_trump: action.data.hsh_wrds_trump,
		hsh_wrds_hillary: action.data.hsh_wrds_hillary,
		popular_twitter_account_references: action.data.popular_twitter_account_references,
		acc_wrds_trump: action.data.acc_wrds_trump,
		acc_wrds_hillary: action.data.acc_wrds_hillary,
		pop_wrds_trump: action.data.pop_wrds_trump,
		pop_wrds_hillary: action.data.pop_wrds_hillary,
		popular_negative_words: action.data.popular_negative_words,
		pw_trump: action.data.pw_trump,
		pw_hillary: action.data.pw_hillary,
		nw_trump: action.data.nw_trump,
		nw_hillary: action.data.nw_hillary,
		sentiment_of_tweets: action.data.sentiment_of_tweets,
		average_retweets: action.data.average_retweets,
		classifier_trump: action.data.classifier_trump,
		classifier_hillary: action.data.classifier_hillary,
		tweets_trump: action.data.tweets_trump,
		tweets_hillary:action.data.tweets_hillary,
		log_reg_trump: action.data.log_reg_trump,
		log_reg_hillary: action.data.log_reg_hillary ,
		svm_trump: action.data.svm_trump,
		svm_hillary: action.data.svm_hillary,
		loading: false,
		predicted: true,
	});
};


const reducer = (state = initialState, action) => {
	switch(action.type) {
		case actionTypes.PROCESSING_START_TEXT_MINING:
			return processingStart(state, action);
		case actionTypes.PREDICT_SUCCESS_TEXT_MINING:
			return predictSuccess(state, action);
		default:
			return state;
	}
};

export default reducer;