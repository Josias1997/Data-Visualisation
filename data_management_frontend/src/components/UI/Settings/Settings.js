import React from "react";
import {MDBBtn, MDBIcon} from "mdbreact";
import { removeFile, reset } from "../../../store/actions";
import { connect } from "react-redux";
import RStatisticsSettings from "./RStatisticsSettings/RStatisticsSettings";
import ModelingSettings from "./ModelingSettings/ModelingSettings";
import DataManipulationSettings from "./DataManipulationSettings/DataManipulationSettings";
import MachineLearningSettings from "./MachineLearningSettings/MachineLearningSettings";
import DashboardSettings from "./Dashboard/DashboardSettings";

const Settings = ({fileId, page, onFileRemove, onPrint, onPredict, onSplit, onReset}) => {
    let content = null;
    switch(page) {
      case 'data-manipulation':
        content = <DataManipulationSettings />;
        break;
      case 'r-statistics':
        content = <RStatisticsSettings />;
        break;
      case 'modeling':
        content = <ModelingSettings />;
        break;
      case 'machine-learning':
        content = <MachineLearningSettings onPrint={onPrint} onPredict={onPredict} onSplit={onSplit} />;
        break;
      case 'dashboard':
        content = <DashboardSettings />;
        break;
    }
    return (
       <div className={"row d-flex justify-content-center"}>
           {content}
          <MDBBtn onClick={onFileRemove}>
              Accueil
          </MDBBtn>
          <MDBBtn color="success" onClick={() => onReset(fileId)}>
            <MDBIcon icon={"sync"} className="mr-2"/>
             Reset
          </MDBBtn>
       </div>
    )
};

const mapStateToProps = state => {
  return {
    fileId: state.fileUpload.id
  }
}

const mapDispatchToProps = dispatch => {
    return {
        onFileRemove: () => dispatch(removeFile()),
        onReset: id => dispatch(reset(id))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Settings);