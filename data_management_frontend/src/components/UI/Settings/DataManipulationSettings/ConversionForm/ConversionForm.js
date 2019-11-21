import React from "react";
import {connect} from "react-redux";
import {applySettings, columnsAndRowsValueChangeHandler} from "../../../../../store/actions";
import {createJsonData} from "../../../../../utility/utility";
import { MDBInputGroup, MDBIcon, MDBBtn, MDBInput} from "mdbreact";


const ConversionForm = (props) => {
    const transformHandler = () => {
        const data = createJsonData(['id', 'column', 'options'], [
            props.fileId, props.columnToTransform, props.options
        ]);
        props.onSettingsApply('/api/transform/', data);
    };
    return (
        <div className={"col-md-4 mt-3"}>
            <MDBInputGroup
                material
                containerClassName="m-0"
                prepend={
                    <MDBBtn className="m-0 px-3 py-2 z-depth-0" onClick={transformHandler}>
                        <MDBIcon icon={"sync"}/> Transformer
                    </MDBBtn>
                }
                inputs={
                  <>
                    <MDBInput noTag type="text" hint="Colonne" id={"columnToTransform"} />
                    <select id={'options'} className="browser-default custom-select"
                        onChange={props.onColumnsAndRowsValueChange}>
                        <option defaultValue={""}>Type conversion</option>
                        <option value="int">Int</option>
                        <option value="float">Float</option>
                        <option value="str">String</option>
                    </select>
                  </>
                }
            />
        </div>
    )
};


const mapStateToProps = state => {
    return {
        fileId: state.fileUpload.id,
        options: state.parameters.options,
        columnToTransform: state.parameters.columnToTransform
    }
};

const mapDispatchToProps = dispatch => {
    return {
        onSettingsApply: (url, data) => dispatch(applySettings(url, data)),
        onColumnsAndRowsValueChange: event => dispatch(columnsAndRowsValueChangeHandler(event))
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(ConversionForm);