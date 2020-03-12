import * as actionTypes from "./actionTypes";

export const predictSuccessDeepLearning = (data) => {
	return {
		type: actionTypes.PREDICT_SUCCESS_DEEP_LEARNING,
		data: data
	}
}

export const processingStartDeepLearning = () => {
	return {
		type: actionTypes.PROCESSING_START_DEEP_LEARNING,
	}
};