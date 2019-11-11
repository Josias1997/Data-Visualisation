import React, { Fragment, useEffect } from 'react';
import Header from './components/Header.jsx';
import LoginPage from './components/Pages/LoginPage.jsx';
import FileForm from './components/UI/FileForm.jsx';
import { connect } from 'react-redux';
import * as actions from './store/actions/auth';

const App = props => {

	useEffect(() => {
		props.onTryAutoSignIn();
	}, [])

	return (
		<Fragment>
			<Header/>
			{
				props.isAuthenticated ? <FileForm /> : <LoginPage />
			}
		</Fragment>
	);
};

const mapStateToProps = state => {
	return {
		isAuthenticated: state.token !== null,
	}
};

const mapDispatchToProps = dispatch => {
	return {
		onTryAutoSignIn: () => dispatch(actions.authCheckState()),
	}
}


export default connect(mapStateToProps, mapDispatchToProps)(App);