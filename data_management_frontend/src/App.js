import React, { useEffect } from 'react';
import Header from './components/Header/Header.js';
import LoginPage from './components/Pages/Login/LoginPage.js';
import SignUpPage from './components/Pages/SignUp/SignUpPage.js';
import { connect } from 'react-redux';
import * as actions from './store/actions/auth';
import { BrowserRouter as Router } from 'react-router-dom';
import Routes from './components/Routes/Routes.js';
import axios from './instanceAxios';
import 'react-notifications/lib/notifications.css';

const App = props => {

	useEffect(() => {
		props.onTryAutoSignIn();
	}, []);

	return (
		<Router>
			<Header/>
			{
				props.isAuthenticated ? <Routes /> : ( props.appState === 'login' ? <LoginPage /> : <SignUpPage /> )
			}
		</Router>
	);
};

const mapStateToProps = state => {
	return {
		isAuthenticated: state.auth.token !== null,
		appState: state.auth.appState
	}
};

const mapDispatchToProps = dispatch => {
	return {
		onTryAutoSignIn: () => dispatch(actions.authCheckState()),
	}
};


export default connect(mapStateToProps, mapDispatchToProps)(App);