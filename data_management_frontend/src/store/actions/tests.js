import * as actionTypes from "./actionTypes";


export const startTest = () => {
    return {
        type: actionTypes.TEST_START
    }
};

export const testSuccess = results => {
    return {
        type: actionTypes.TEST_SUCCESS,
        results: results
    }
};

export const testFail = error => {
    return {
        type: actionTypes.TEST_FAIL,
        error: error,
    }
};

export const test = (data) => {
    return dispatch => {
        dispatch(startTest());
        
    }
};