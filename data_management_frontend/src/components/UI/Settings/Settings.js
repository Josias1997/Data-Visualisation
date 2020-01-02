import React from "react";
import {MDBBtn, MDBIcon} from "mdbreact";
import { removeFile } from "../../../store/actions";
import { connect } from "react-redux";
import RStatisticsSettings from "./RStatisticsSettings/RStatisticsSettings";
import ModelingSettings from "./ModelingSettings/ModelingSettings";
import DataManipulationSettings from "./DataManipulationSettings/DataManipulationSettings";
import MachineLearningSettings from "./MachineLearningSettings/MachineLearningSettings";

const Settings = ({page, onFileRemove, onFit, onPredict, onSplit}) => {
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
        content = <MachineLearningSettings onFit={onFit} onPredict={onPredict} onSplit={onSplit} />;
        break;
    }
    return (
       <div className={"row d-flex justify-content-center"}>
           {content}
           <div className={"col-md-2"}>
                <MDBBtn color="danger" onClick={onFileRemove}>
                    <MDBIcon icon={"trash"}/>
                </MDBBtn>
           </div>
       </div>
    )
};

const mapDispatchToProps = dispatch => {
    return {
        onFileRemove: () => dispatch(removeFile()),
    }
}

export default connect(null, mapDispatchToProps)(Settings);