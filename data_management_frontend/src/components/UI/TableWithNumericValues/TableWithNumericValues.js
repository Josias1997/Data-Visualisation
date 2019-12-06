import React from "react";
import { MDBTable, MDBTableBody, MDBTableHead } from 'mdbreact';

const TableWithNumericValues = ({data}) => {
	return (
		<MDBTable bordered>
      <MDBTableBody>
      {
      	data.map((row, index) => {
      		return (
      			<tr>
      				<td>{index + 1}</td>
      				{
      					row.map(value => <td>
      						{value}
      					</td>)
      				}
      			</tr>
      		)
      	})
      }
      </MDBTableBody>
    </MDBTable>
	)
};

export default TableWithNumericValues;