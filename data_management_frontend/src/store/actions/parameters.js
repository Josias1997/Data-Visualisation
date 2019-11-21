import * as actionTypes from './actionTypes';
import axios from '../../instanceAxios';
import {updateDataStart, updateDataSuccess, updateDataFail} from "./fileUpload";


export const columnInputValueChanged = target => {
    return {
        type: actionTypes.COLUMN_INPUT_VALUE_CHANGED,
        target: target
    }
};

export const paramsForFilteringChanged = target => {
    return {
        type: actionTypes.PARAMS_FOR_FILTERING_CHANGED,
        target: target
    }
};

export const applySettings = (url, data) => {
    return dispatch => {
        dispatch(updateDataStart());
        axios.post(url, data)
            .then(response => {
                dispatch(updateDataSuccess(response.data))
            })
            .catch(error => {
                dispatch(updateDataFail(error))
            })
    }
};

export const handleColumnsChange = (event) => {
    const target = event.target;
    return dispatch => {
        dispatch(columnInputValueChanged(target))
    };
};

export const columnsAndRowsValueChangeHandler = event => {
        return dispatch => {
            dispatch(paramsForFilteringChanged(event.target));
        };
};