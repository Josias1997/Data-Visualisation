import React from 'react';
import Statistics from "./Statictics/Statistics";
import SQLQueryForm from "./SQLQueryForm/SQLQueryForm";
import ConversionForm from "./ConversionForm/ConversionForm";
import { MDBBtn, MDBIcon } from 'mdbreact';
import { connect } from 'react-redux';
import { getInfos } from './../../../../store/actions';

const DataManipulationSettings = ({fileId, onInfos}) => {
    return (
        <>
            <Statistics />
            <SQLQueryForm/>
            <MDBBtn color="success" onClick={() => onInfos(fileId)}>
                <MDBIcon icon={"sync"}/>
                Missing Values
            </MDBBtn>
        </>
    );
}

const mapStateToProps = state => {
  return {
    fileId: state.fileUpload.id
  }
}

const mapDispatchToProps = dispatch => {
    return {
        onInfos: id => dispatch(getInfos(id))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(DataManipulationSettings);