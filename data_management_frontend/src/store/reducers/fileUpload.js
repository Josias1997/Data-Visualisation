import * as actionTypes from './../actions/actionTypes';
import {formatErrorData, updateObject} from "../../utility/utility";

const initialState = {
    name: '',
    file: undefined,
    file_data: undefined,
    id: undefined,
    loading: false,
    error: null,
    columnsNames: [],
    size: 0,
    rows: 0,
    columns: 0
};

const uploadStart = (state, action) => {
    return updateObject(state, {
        loading: true,
        error: null,
    })
};

const uploadSuccess = (state, action) => {
    return updateObject(state, {
        name: action.details.name,
        file_data: action.details.data,
        id: action.details.id,
        loading: false,
        error: null,
        columnsNames: action.details.columnsNames,
        size: action.details.size,
        rows: action.details.rows,
        columns: action.details.columns,
    })
};

const uploadFail = (state, action) => {
    return updateObject(state, {
        loading: false,
        file_data: formatErrorData('Erreur lors du chargement du fichier veuiller reprendre.'),
        error: action.error,
    })
};

const inputValueChanged = (state, action) => {
    return updateObject(state, {
        file: action.file,
        loading: false,
        error: null
    })
};

const updateDataStart = (state, action) => {
    return updateObject(state, {
        loading: true,
        error: null
    })
};

const updateDataSuccess = (state, action) => {
    return updateObject(state, {
        file_data: action.data,
        loading: false,
        error: null,
    })
};

const updateDataFail = (state, action) => {
    return updateObject(state, initialState);
};

const removeFile = (state, action) => {
    return updateObject(state, initialState);
}

const reducer = (state = initialState, action) => {
    switch(action.type) {
        case actionTypes.UPLOAD_START:
            return uploadStart(state, action);
        case actionTypes.UPLOAD_FAIL:
            return uploadFail(state, action);
        case actionTypes.UPLOAD_SUCCESS:
            return uploadSuccess(state, action);
        case actionTypes.INPUT_VALUE_CHANGED:
            return inputValueChanged(state, action);
        case actionTypes.UPDATE_DATA_START:
            return updateDataStart(state, action);
        case actionTypes.UPDATE_DATA_SUCCESS:
            return updateDataSuccess(state, action);
        case actionTypes.UPDATE_DATA_FAIL:
            return updateDataFail(state, action);
        case actionTypes.REMOVE_FILE:
            return removeFile(state, action);
        default:
            return state;
    }
};

export default reducer;