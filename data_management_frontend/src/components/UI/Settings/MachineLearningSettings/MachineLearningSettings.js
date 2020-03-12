import React, { useState } from "react";
import { MDBBtn, MDBIcon } from "mdbreact";
import { connect } from "react-redux";
import Radium from 'radium';

const MachineLearningSettings = props => {
	const [visible, setVisible] = useState(false);

	const exportData = (type) => {
		props.onPrint(type);
	};

	const style = {
		cursor: 'pointer',
		':hover': {
			background: 'lightgreen'
		}
	};

	return (
		<>
		<div className="col-md-8">
			<MDBBtn onClick = {() => props.onPredict()} >
				<MDBIcon icon="brain" className="mr-2" />
				Predict
			</MDBBtn>
			<MDBBtn onClick = {() => props.onSplit()}>
				<MDBIcon icon="pen-nib" className="mr-2" />
				Split Data Set
			</MDBBtn>
			{
				props.predicted ? <MDBBtn onClick = {() => setVisible(current => !current)}>
					<MDBIcon icon="download" className="mr-2" />
					Export
				</MDBBtn> : null
			}
		</div>
			{
				visible ? <div style={{
					position: 'absolute',
					zIndex: 100,
					width: '25%',
					background: 'white',
					top: '94%',
					padding: '10px',
					border: '1px solid gray'
				}}>
					<div key="pdf" style={style} onClick={() => exportData("pdf")}>PDF</div>
					<div key="jpeg" style={style} onClick={() => exportData("jpeg")}>JPEG</div>
					<div key="png" style={style} onClick={() => exportData("png")}>PNG</div>
				</div> : null
			}
		</>
	)
};

const mapStateToProps = state => {
  return {
      predicted: state.machine_learning.predicted || state.deep_learning.predicted
  }
};


export default connect(mapStateToProps)(Radium(MachineLearningSettings));