import * as actionTypes from "./actionTypes";
import axios from "../../instanceAxios";

export const startPreprocessing = () => {
	return {
		type: actionTypes.PREPROCESSING_START,
	}
};

export const preprocessingDataSplitSuccess = (data) => {
	return {
		type: actionTypes.PREPROCESSING_DATA_SPLIT_SUCCESS,
		data: data
	}
};

export const preprocessingNormalizingSuccess = (data) => {
	return {
		type: actionTypes.PREPROCESSING_NORMALIZING_SUCCESS,
		data: data
	}
}

export const preprocessingFail = (error) => {
	return {
		type: actionTypes.PREPROCESSING_FAIL,
		error: error
	}
};

export const splitDataSet = (data) => {
	return dispatch => {
		dispatch(startPreprocessing());
		axios.post('/api/split-data-set/', data)
		.then(({data}) => {
			dispatch(preprocessingDataSplitSuccess(data));
		}).catch(error => {
			dispatch(preprocessingFail(error.message));
		})
	}
};

export const normalize = (data) => {
	return dispatch => {
		dispatch(startPreprocessing());
		axios.post('/api/preprocessing/', data)
		.then(({data}) => {
			dispatch(preprocessingNormalizingSuccess(data));
		}).catch(error => {
			dispatch(preprocessingFail(error.message));
		})	
	}
};