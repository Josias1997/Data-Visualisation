import * as actionTypes from "./actionTypes";

export const predictSuccessTextMining = (data) => {
	return {
		type: actionTypes.PREDICT_SUCCESS_TEXT_MINING,
		data: data
	}
};

export const processingStartTextMining = () => {
	return {
		type: actionTypes.PROCESSING_START_TEXT_MINING,
	}
};