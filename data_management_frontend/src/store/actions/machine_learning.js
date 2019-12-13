import * as actionTypes from "./actionTypes";
import axios from "../../instanceAxios";


export const processingStart = () => {
	return {
		type: actionTypes.PROCESSING_START
	}
};

export const fitSuccess = (data) => {
	return {
		type: actionTypes.FIT_SUCCESS,
		data: data
	}
};

export const predictSuccess = (data) => {
	return {
		type: actionTypes.PREDICT_SUCCESS,
		data: data
	}
}

export const processingFail = (error) => {
	return {
		type: actionTypes.PROCESSING_FAIL,
		error: error
	}
};

export const fit = (data) => {
	return dispatch => {
		dispatch(processingStart());
		axios.post('/api/fit-data-set/', data)
		.then(({data}) => {
			dispatch(fitSuccess(data));
		}).catch(error => {
			console.log(data);
			dispatch(processingFail(error.message));
		})
	}
}

export const predict = (data) => {
	return dispatch => {
		dispatch(processingStart());
		axios.post('/api/predict-data-set/', data)
		.then(({data}) => {
			console.log(data)
			dispatch(predictSuccess(data));
		}).catch(error => {
			dispatch(processingFail(error.message));
		})
	}
}

