import * as actionTypes from "./actionTypes";
import axios from "../../instanceAxios";
import { predictSuccessDeepLearning, processingStartDeepLearning } from './deep_learning';
import { predictSuccessTextMining, processingStartTextMining } from './text_mining';


export const processingStart = () => {
	return {
		type: actionTypes.PROCESSING_START_MACHINE_LEARNING,
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
		type: actionTypes.PREDICT_SUCCESS_MACHINE_LEARNING,
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

export const predict = (data, from) => {
	return dispatch => {
		switch(from) {
			case 'deep_learning':
				dispatch(processingStartDeepLearning());
				break;
			case 'text_mining':
				dispatch(processingStartTextMining());
				break;
			default:
				dispatch(processingStart());
				break;
		}
		axios.post('/api/predict-data-set/', data)
		.then(({data}) => {
			console.log(data)
			switch(from) {
				case 'deep_learning':
					dispatch(predictSuccessDeepLearning(data));
					break;
				case 'text_mining':
					dispatch(predictSuccessTextMining(data));
					break;
				default:
					dispatch(predictSuccess(data));
					break;
			}
		}).catch(error => {
			dispatch(processingFail(error.message));
		})
	}
}

