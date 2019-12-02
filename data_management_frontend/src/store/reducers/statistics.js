import { updateObject } from "../../utility/utility";
import * as actionTypes from '../actions/actionTypes';

const initialState = {
    openTable: true,
    openPlot: false,
    openTests: false
};

const openPlot = (state) => {
    return updateObject(state, {
        openTable: false,
        openPlot: true,
        openTests: false
    });
};

const openTable = (state) => {
    return updateObject(state, initialState);
};

const openTests = (state) => {
    return updateObject(state, {
        openTable: false,
        openPlot: false,
        openTests: true
    });
};

const reducer = (state = initialState, action) => {
    switch(action.type) {
        case actionTypes.OPEN_TABLE:
            return openTable(state);
        case actionTypes.OPEN_PLOT:
            return openPlot(state);
        case actionTypes.OPEN_TESTS:
            return openTests(state);
        default:
            return state;
    }
};

export default reducer;