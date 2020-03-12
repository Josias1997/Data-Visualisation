import * as actionTypes from '../actions/actionTypes';
import { updateObject } from '../../utility/utility';

const initialState = {
	appState: 'login',
	token: null,
	error: null,
	loading: false,
};


const authStart = (state, action) => {
	return updateObject(state, {
		error: null,
		loading: true,
	})
};

const authSuccess = (state, action) => {
	return updateObject(state, {
		token: action.token,
		error: null,
		loading: false,
	})
};

const authFail = (state, action) => {
	return updateObject(state, {
		error: action.error,
		loading: false,
	})
};

const authLogOut = (state, action) => {
	return updateObject(state, {
		token: null,
	})
};

const login = (state, action) => {
	return updateObject(state, initialState)
}

const register = (state, action) => {
	return updateObject(state, {
		appState: 'register'
	})
}

const reducer = (state=initialState, action) => {
	switch(action.type) {
		case actionTypes.AUTH_START:
			return authStart(state, action);
		case actionTypes.AUTH_SUCCESS:
			return authSuccess(state, action);
		case actionTypes.AUTH_FAIL:
			return authFail(state, action);
		case actionTypes.AUTH_LOGOUT:
			return authLogOut(state, action);
		case actionTypes.LOGIN_STATE:
			return login(state, action);
		case actionTypes.REGISTER_STATE:
			return register(state, action);
		default:
			return state;
	}
}

export default reducer;