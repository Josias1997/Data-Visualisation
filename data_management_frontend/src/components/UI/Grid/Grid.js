import React from "react";
import {MDBCol, MDBRow} from "mdbreact";

const Grid = ({children}) => {
    return (
        <MDBRow>
            <MDBCol md={"3"}>

            </MDBCol>
            <MDBCol md={"6"}>
                {children}
            </MDBCol>
            <MDBCol md={"3"}>

            </MDBCol>
        </MDBRow>
    )
};

export default Grid;