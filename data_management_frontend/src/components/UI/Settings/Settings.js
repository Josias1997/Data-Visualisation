import React from "react";
import {MDBBtn, MDBIcon} from "mdbreact";
import { removeFile } from "../../../store/actions";
import { connect } from "react-redux";
import RStatisticsSettings from "./RStatisticsSettings/RStatisticsSettings";
import ModelingSettings from "./ModelingSettings/ModelingSettings";
import DataManipulationSettings from "./DataManipulationSettings/DataManipulationSettings";

const Settings = ({page, onFileRemove}) => {
    let content = null;
    if (page === 'data-manipulation') {
        content = <DataManipulationSettings />;
    }
    else if (page === 'r-statistics') {
        content = <RStatisticsSettings />;
    }
    else if (page === 'modeling') {
        content = <ModelingSettings />
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