import React, { Component } from "react";

class ErrorBoundary extends Component {
	constructor(props) {
		super(props);
		this.state = {
			hasError: false
		};
	};

	static getDerivedStateFromError(error) {
		return {hasError: true}
	}

	componentDidCatch(error, errorInfo) {
		console.log(error, errorInfo);
	}

	render () {
		return this.state.hasError ? <h1>Something went wrong</h1> : this.props.children;
	}
};

export default ErrorBoundary;