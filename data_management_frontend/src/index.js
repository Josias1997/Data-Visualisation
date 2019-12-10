import React from 'react';
import ReactDOM from 'react-dom';
import App from './App.js';
import authReducer from './store/reducers/auth';
import fileUploadReducer from './store/reducers/fileUpload';
import parametersReducer from './store/reducers/parameters';
import statisticsReducer from './store/reducers/statistics';
import testsReducer from './store/reducers/tests';
import modelisationReducer from './store/reducers/modelisation';
import machineLearningReducer from './store/reducers/machine_learning';
import { Provider } from 'react-redux';
import { createStore, compose, applyMiddleware, combineReducers } from 'redux';
import thunk from 'redux-thunk';

let rootReducer = combineReducers({
	auth: authReducer,
	fileUpload: fileUploadReducer,
	parameters: parametersReducer,
	statistics: statisticsReducer,
	tests: testsReducer,
	modelisation: modelisationReducer,
	machine_learning: machineLearningReducer,
});

const composeEnhancers = process.env.NODE_ENV === 'development' ? window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ : null || compose;
const store = createStore(rootReducer, composeEnhancers(
	applyMiddleware(thunk)
));

ReactDOM.render(<Provider store={store}>
	<App/>
</Provider>, document.getElementById("root"));