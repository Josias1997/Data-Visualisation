import * as actionTypes from './actionTypes';
import axios from '../../instanceAxios';


export const authStart = () => {
	return {
		type: actionTypes.AUTH_START,
	}
};

export const authSuccess = token => {
	return {
		type: actionTypes.AUTH_SUCCESS,
		token: token,
	}
};
export const authFail = error => {
	return {
		type: actionTypes.AUTH_FAIL,
		error: error,
	}
};
export const logout = () => {
	localStorage.removeItem('token');
	localStorage.removeItem('expirationDate');
	return {
		type: actionTypes.AUTH_LOGOUT
	}
};

export const checkAuthTimeOut = expirationTime => {
	return dispatch => {
		setTimeout(() => {
			dispatch(logout());
		}, expirationTime * 1000)
	}
}

const generateSession = (response, dispatch) => {
	const token = response.data.key;
	console.log(token);
	const expirationDate = new Date(new Date().getTime() + 3600 * 1000);
	localStorage.setItem('token', token);
	localStorage.setItem('expirationDate', expirationDate);
	dispatch(authSuccess(token));
	dispatch(checkAuthTimeOut(3600));
}


export const authLogin = (username, password) => {
	return dispatch => {
		dispatch(authStart());
		const data = {
			username: username,
			password: password
		};
		axios.post('/api/login/', data, {
			headers: {
				"Content-Type": 'application/json'
			}
		})
		.then(response => {
			generateSession(response, dispatch);
		})
		.catch(error => {
			dispatch(authFail(error))
		});
	}
};

/* Ajout Inscription 
export const authSignup = (username, email, password, passwordConfirmed) => {
	return dispatch => {
		dispatch(authStart());
		axios.post('/rest-auth/registration/', {
			username: username,
			email: email,
			passwor1: password,
			password2: passwordConfirmed,
		})
		.then(response => {
			generateSession(response, dispatch);
		})
		.catch(error => {
			dispatch(authFail(error))
		});
	}
};
*/

export const authCheckState = () => {
	return dispatch => {
		const token = localStorage.getItem('token');
		if (token === undefined) {
			dispatch(logout());
		} else {
			const expirationDate = new Date(localStorage.getItem('expirationDate'));
			if ( expirationDate <= new Date()) {
				dispatch(logout())
			} else {
				dispatch(authSuccess(token));
				dispatch(checkAuthTimeOut((expirationDate.getTime() - new Date().getTime()) / 1000));
			}
		}
	}
}
