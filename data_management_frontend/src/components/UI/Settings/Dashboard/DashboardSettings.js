import React from "react";
import { MDBBtn, MDBIcon } from "mdbreact";
import { connect } from "react-redux";
import { openTable, openPlot, openStorytelling, openDashboard } from '../../../../store/actions';
import { createJsonData } from "../../../../utility/utility";

const DashboardSettings = (props) => {

  return (
    <div className="col-md-8">
      <MDBBtn color={props.openTable ? "primary" : "secondary"} onClick={props.handleOpenTable}>
        <MDBIcon icon={"table"} className="mr-1"/>
        Table
      </MDBBtn>
      <MDBBtn color={props.openPlot ? "primary" : "secondary"} onClick={props.handleOpenPlot}>
        <MDBIcon icon={"chart-pie"} className="mr-1"/>
        Plots
      </MDBBtn>
      <MDBBtn color={props.openDashboard ? "primary" : "secondary"} onClick={props.handleOpenDashboard}>
        <MDBIcon icon={"chart-pie"} className="mr-1"/>
        Dashboard
      </MDBBtn>
      <MDBBtn color={props.openStorytelling ? "primary" : "secondary"} onClick={props.handleOpenStorytelling}>
        <MDBIcon icon={"pen"} className="mr-1"/>
        Storytelling
      </MDBBtn>
    </div>
  );
};

const mapStateToProps = state => {
  return {
      id: state.fileUpload.id,
      openPlot: state.statistics.openPlot,
      openTable: state.statistics.openTable,
      openStorytelling: state.statistics.openStorytelling,
      openDashboard: state.statistics.openDashboard,
  }
};

const mapDispatchToProps = dispatch => {
  return {
      handleOpenTable: () => dispatch(openTable()),
      handleOpenPlot: () => dispatch(openPlot()),
      handleOpenStorytelling: () => dispatch(openStorytelling()),
      handleOpenDashboard: () => dispatch(openDashboard())
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(DashboardSettings);