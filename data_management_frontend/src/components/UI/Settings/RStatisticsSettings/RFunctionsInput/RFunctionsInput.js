import React from "react";
import { MDBBtn, MDBIcon } from "mdbreact";
import { connect } from "react-redux";
import { openTable, openPlot, openTests, openStats, applySettings } from '../../../../../store/actions';
import {createJsonData} from "../../../../../utility/utility";

const RFunctionsInput = (props) => {

  const describe = () => {
      const data = createJsonData(['id'], [props.id]);
      props.onDescribe('/api/describe/', data);
      props.handleOpenStats();
  };
  return (
    <div className="col-md-8">
      <MDBBtn color={props.openTable ? "primary" : "secondary"} onClick={props.handleOpenTable}>
        <MDBIcon icon={"table"} className="mr-1"/>
        Table
      </MDBBtn>
      <MDBBtn color={props.openTests ? "primary" : "secondary"} onClick={props.handleOpenTests}>
        <MDBIcon icon={"question"} className="mr-1"/>
        Tests
      </MDBBtn>
      <MDBBtn color="default" onClick={describe}>
        <MDBIcon icon="chart-line" className="mr-1"/> 
        Stats
      </MDBBtn>
      <MDBBtn color={props.openPlot ? "primary" : "secondary"} onClick={props.handleOpenPlot}>
        <MDBIcon icon={"chart-pie" } className="mr-1"/>
        Plot
      </MDBBtn>
    </div>
  );
};

const mapStateToProps = state => {
  return {
      id: state.fileUpload.id,
      openPlot: state.statistics.openPlot,
      openTable: state.statistics.openTable,
      openTests: state.statistics.openTests,
      openStats: state.statistics.openStats,
  }
};

const mapDispatchToProps = dispatch => {
  return {
      handleOpenTable: () => dispatch(openTable()),
      handleOpenPlot: () => dispatch(openPlot()),
      handleOpenTests: () => dispatch(openTests()),
      handleOpenStats: () => dispatch(openStats()),
      onDescribe: (url, data) => dispatch(applySettings(url, data))
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(RFunctionsInput);