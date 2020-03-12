import * as actionTypes from "./../actions/actionTypes";
import { updateObject } from "../../utility/utility";

const initialState = {
	loading: false,
	error: false,
	predicted: false,
	confusionMatrix: [],
	matrixPlot: '',
	stockPricePlot: '',
	lstmPlot: '',
	gruPlot: '',
	sequencePlot: '',
	samples1: '',
	samples2: '',
	examples: '',
	model: '',
	matrixPlot: '',

}

const processingStart = (state) => {
	return updateObject(state, {
		loading: true,
	});
};


const predictSuccess = (state, action) => {
	return updateObject(state, {
		confusionMatrix: action.data.confusion_matrix,
		matrixPlot: action.data.matrix_plot,
		stockPricePlot: action.data.stock_price_plot,
		lstmPlot: action.data.lstm_plot,
		gruPlot: action.data.gru_plot,
		sequencePlot: action.data.sequence_plot,
		samples1: action.data.samples_1,
		samples2: action.data.samples_2,
		examples: action.data.examples,
		model: action.data.model,
		matrixPlot: action.data.confusion_matrix,
		loading: false,
		predicted: true,
	});
};


const reducer = (state = initialState, action) => {
	switch(action.type) {
		case actionTypes.PROCESSING_START_DEEP_LEARNING:
			return processingStart(state, action);
		case actionTypes.PREDICT_SUCCESS_DEEP_LEARNING:
			return predictSuccess(state, action);
		default:
			return state;
	}
};

export default reducer;