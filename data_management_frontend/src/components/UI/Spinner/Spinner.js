import React from 'react';

const Spinner = props => {
	return (
		<div className={"d-flex justify-content-center"}>
		<div className="spinner-border mt-5" role="status">
	        <span className="sr-only align-middle">Loading...</span>
	    </div>
		</div>
	)
};

export default Spinner;