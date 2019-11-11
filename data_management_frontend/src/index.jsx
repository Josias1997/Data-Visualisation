import React from 'react';
import ReactDOM from 'react-dom';
import App from './App.jsx';
import reducer from './store/reducers/auth';
import { Provider } from 'react-redux';
import { createStore, compose, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';


const composeEnhancers = process.env.NODE_ENV === 'development' ? window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ : null || compose;
const store = createStore(reducer, composeEnhancers(
	applyMiddleware(thunk)
));

ReactDOM.render(<Provider store={store}>
	<App/>
</Provider>, document.getElementById("root"));