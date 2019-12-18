import React from 'react';
import { MDBCol, MDBRow, MDBContainer} from 'mdbreact';

const Matrix = props => {
	return (
		<div className="col-md-9">
			<MDBRow>
				<MDBCol col={4}>
				</MDBCol>
				<MDBCol>
				  <MDBRow>
				  	<MDBCol col={2}>
				   		<h3>PREDICTIVE VALUES</h3>
				   </MDBCol>
				  </MDBRow>
				  <MDBRow>
					<MDBCol col={1}>
				  		<h3>Positive (1)</h3>
				  	</MDBCol>
				  	<MDBCol col={1}>
				  		<h3>Negative (0)</h3>
				  	</MDBCol>
				  </MDBRow>
				</MDBCol>
			</MDBRow>
			<MDBRow>
				<MDBCol col={1} style={{
						direction: 'ltr',
						textOrientation: 'vertical-rl',
					}}>
					<h3>ACTUAL VALUES</h3>
				</MDBCol>
				<MDBCol col={2}>
				 <MDBRow>
					 <MDBCol col={1}>
					 	<h3>Positive (1)</h3>
					 </MDBCol>
				 </MDBRow>
				 <MDBRow>
				 	<MDBCol col={1}><h3>Negative (0)</h3></MDBCol>
				 </MDBRow>
				</MDBCol>
				<MDBCol col={2}>
				{
					props.matrix.map((arr, index) => (<MDBRow key={index}>
							{
								arr.map((value, index) => (
									<MDBCol>
										<h3>{value}</h3>
									</MDBCol>
								))
							}
						</MDBRow>))
				}
				</MDBCol>
			</MDBRow>
		</div>
	)
}

export default Matrix;
