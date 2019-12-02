import * as actionTypes from "../actions/actionTypes";
import { updateObject } from "../../utility/utility";

const initialState = {
    result: '',
    error: false,
    loading: false
}

const startTest = (state) => {
    return updateObject(state, {
        result: '',
        loading: true,
        error: false,
    })
};

const testSuccess = (state, action) => {
    return updateObject(state, {
        result: action.result,
        loading: false,
        error: false
    })
};

const testFail = (state, action) => {
    return updateObject(state, {
        result: '',
        loading: false,
        error: action.error
    })
};

const reducer = (state = initialState, action) => {
    switch(action.type) {
        case actionTypes.TEST_START:
            return startTest(state);
        case actionTypes.TEST_SUCCESS:
            return testSuccess(state, action);
        case actionTypes.TEST_FAIL:
            return testFail(state, action);
        default:
            return state;
    }
};

export default reducer;