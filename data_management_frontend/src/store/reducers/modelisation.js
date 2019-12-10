import { updateObject } from "../../utility/utility";
import * as actionTypes from "../actions/actionTypes";

const initialState = {
	trainingSet: {},
	testSet: {},
	processing: false,
	error: false,
	normalized: false,
	normalizedTrainingSet: [],
};

const startPreprocessing = (state) => {
	return updateObject(state, {
		processing: true,
	})
};

const preprocessingDataSplitSuccess = (state, action) => {
	return updateObject(state, {
		trainingSet: action.data.training_set,
		testSet: action.data.test_set,
		error: action.data.error,
		processing: false
	})
};

const preprocessingNormalizingSuccess = (state, action) => {
	return updateObject(state, {
		normalizedTrainingSet: action.data.normalized_training_set,
		error: action.data.error,
		normalized: true,
		processing: false
	})
};

const preprocessingFail = (state, action) => {
	return updateObject(state, {
		error: action.data.error,
		processing: false,
	})
};

const reducer = (state = initialState, action) => {
	switch(action.type) {
		case actionTypes.PREPROCESSING_START:
			return startPreprocessing(state);
		case actionTypes.PREPROCESSING_DATA_SPLIT_SUCCESS:
			return preprocessingDataSplitSuccess(state, action);
		case actionTypes.PREPROCESSING_NORMALIZING_SUCCESS:
			return preprocessingNormalizingSuccess(state, action);
		case actionTypes.PREPROCESSING_FAIL:
			return preprocessingFail(state, action);
		default:
			return state;
	}
};

export default reducer;