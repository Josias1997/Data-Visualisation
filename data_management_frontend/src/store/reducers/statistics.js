import { updateObject } from "../../utility/utility";
import * as actionTypes from '../actions/actionTypes';

const initialState = {
    openTable: true,
    openPlot: false,
    openTests: false,
    openStats: false,
    openStorytelling: false,
    openDashboard: false
};

const openPlot = (state) => {
    return updateObject(state, {
        openTable: false,
        openPlot: true,
        openTests: false,
        openStats: false,
        openStorytelling: false,
        openDashboard: false
    });
};

const openTable = (state) => {
    return updateObject(state, initialState);
};

const openTests = (state) => {
    return updateObject(state, {
        openTable: false,
        openPlot: false,
        openTests: true,
        openStats: false,
        openStorytelling: false,
        openDashboard: false
    });
};

const openStats = (state) => {
    return  updateObject(state, {
        openTable: false,
        openPlot: false,
        openTests: false,
        openStats: true,
        openStorytelling: false,
        openDashboard: false
    })
};

const openStorytelling = (state) => {
    return updateObject(state, {
        openTable: false,
        openPlot: false,
        openTests: false,
        openStats: false,
        openStorytelling: true,
        openDashboard: false
    })
};

const openDashboard = (state) => {
    return updateObject(state, {
        openTable: false,
        openPlot: false,
        openTests: false,
        openStats: false,
        openStorytelling: false,
        openDashboard: true
    })
}



const reducer = (state = initialState, action) => {
    switch(action.type) {
        case actionTypes.OPEN_TABLE:
            return openTable(state);
        case actionTypes.OPEN_PLOT:
            return openPlot(state);
        case actionTypes.OPEN_TESTS:
            return openTests(state);
        case actionTypes.OPEN_STATS:
            return openStats(state);
        case actionTypes.OPEN_STORYTELLING:
            return openStorytelling(state);
        case actionTypes.OPEN_DASHBOARD:
            return openDashboard(state);
        default:
            return state;
    }
};

export default reducer;