import React from "react";
import { MDBTable, MDBTableHead, MDBTableBody } from "mdbreact";

const Table = ({data}) => {
    return (
        <MDBTable autoWidth responsive scrollY>
            <MDBTableHead columns={data.columns} />
            <MDBTableBody rows={data.rows} />
        </MDBTable>
    );
};

export default Table;