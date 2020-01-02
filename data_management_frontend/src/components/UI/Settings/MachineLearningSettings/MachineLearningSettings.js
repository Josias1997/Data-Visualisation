import React from "react";
import { MDBBtn, MDBIcon } from "mdbreact";


const MachineLearningSettings = props => {
	return (
		<div className="col-md-8">
			<MDBBtn onClick = {() => props.onPredict()} >
				<MDBIcon icon="brain" className="mr-2" />
				Predict
			</MDBBtn>
			<MDBBtn onClick = {() => props.onSplit()}>
				<MDBIcon icon="pen-nib" className="mr-2" />
				Split Data Set
			</MDBBtn>
		</div>
	)
};


export default MachineLearningSettings;