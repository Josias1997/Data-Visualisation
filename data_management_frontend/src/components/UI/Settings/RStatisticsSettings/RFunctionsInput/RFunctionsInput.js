import React from "react";
import { MDBBtn, MDBIcon } from "mdbreact";
import { connect } from "react-redux";
import { openTable, openPlot, openTests } from '../../../../../store/actions';

const RFunctionsInput = (props) => {
  return (
    <div className="col-md-8">
      <MDBBtn color={props.openPlot ? "primary" : "secondary"} onClick={props.handleOpenPlot}>
        <MDBIcon icon={"chart-pie" } className="mr-1"/>
        Plot
      </MDBBtn>
      <MDBBtn color={props.openTable ? "primary" : "secondary"} onClick={props.handleOpenTable}>
        <MDBIcon icon={"table"} className="mr-1"/>
        Table
      </MDBBtn>
      <MDBBtn color={props.openTests ? "primary" : "secondary"} onClick={props.handleOpenTests}>
        <MDBIcon icon={"question"} className="mr-1"/>
        Tests
      </MDBBtn>
    </div>
  );
};

const mapStateToProps = state => {
  return {
      id: state.fileUpload.id,
      openPlot: state.statistics.openPlot,
      openTable: state.statistics.openTable,
      openTests: state.statistics.openTests
  }
};

const mapDispatchToProps = dispatch => {
  return {
      handleOpenTable: () => dispatch(openTable()),
      handleOpenPlot: () => dispatch(openPlot()),
      handleOpenTests: () => dispatch(openTests()),
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(RFunctionsInput);