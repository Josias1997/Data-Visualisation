import * as actionTypes from './../actions/actionTypes';
import {updateObject} from "../../utility/utility";

const initialState = {
    beginLine: '',
    endLine: '',
    beginColumn: '',
    endColumn: '',
    searchInput: '',
    columnToTransform: '',
    options: '',
    query: '',
    selectedColumns: [],
};

const updateInputValues = (state, action) => {
    let id = action.target.id;
    let value = action.target.value;
    let newState = {...state};
    newState[id] = value;
    return updateObject(state, newState);
};

const updateCurrentColumns = (state, action) => {
    let isChecked = action.target.checked;
    let id = action.target.id;
    let currentColumns = [...state.selectedColumns];
    if (isChecked) {
        currentColumns.push(id);
    } else {
        if (currentColumns.includes(id)) {
            currentColumns.splice(currentColumns.indexOf(id), 1);
        }
    }
    return updateObject(state, {
        selectedColumns: currentColumns,
    })
};

const reducer = (state = initialState, action) => {
    switch(action.type) {
        case actionTypes.PARAMS_FOR_FILTERING_CHANGED:
            return updateInputValues(state, action);
        case actionTypes.COLUMN_INPUT_VALUE_CHANGED:
            return updateCurrentColumns(state, action);
        default:
            return state;
    }
};

export default reducer;