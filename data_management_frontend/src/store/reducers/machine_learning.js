import * as actionTypes from "./../actions/actionTypes";
import { updateObject } from "../../utility/utility";

const initialState = {
	fitResult: [],
	predictResult: [],
	loading: false,
	error: false,
	predicted: false,
	trainPlotPath: '',
	testPlotPath: '',
}

const processingStart = (state) => {
	return updateObject(state, {
		loading: true,
	});
};

const fitSuccess = (state, action) => {
	return updateObject(state, {
		fitResult: action.data.fit_result,
		loading: false
	});
};

const predictSuccess = (state, action) => {
	return updateObject(state, {
		predictResult: action.data.predict_result,
		trainPlotPath: action.data.train_plot,
		testPlotPath: action.data.test_plot,
		loading: false,
		predicted: true,
	});
};

const processingFail = (state, action) => {
	return updateObject(state, {
		error: action.error,
		loading: false
	})
}

const reducer = (state = initialState, action) => {
	switch(action.type) {
		case actionTypes.PROCESSING_START:
			return processingStart(state);
		case actionTypes.FIT_SUCCESS:
			return fitSuccess(state, action);
		case actionTypes.PREDICT_SUCCESS:
			return predictSuccess(state, action);
		case actionTypes.PROCESSING_FAIL:
			return processingFail(state, action);
		default:
			return state;
	}
};

export default reducer;