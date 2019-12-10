import React from "react";
import { MDBBtn, MDBIcon } from "mdbreact";


const MachineLearningSettings = props => {
	return (
		<div className="col-md-8">
			<MDBBtn onClick = {() => props.onPredict()} >
				<MDBIcon icon="brain" className="mr-2" />
				Predict
			</MDBBtn>
		</div>
	)
};


export default MachineLearningSettings;