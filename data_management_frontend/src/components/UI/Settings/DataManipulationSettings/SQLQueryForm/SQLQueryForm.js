import React from "react";
import {applySettings, columnsAndRowsValueChangeHandler} from "../../../../../store/actions";
import {connect} from "react-redux";
import {createJsonData} from "../../../../../utility/utility";
import {MDBInputGroup, MDBBtn, MDBIcon, MDBInput} from "mdbreact";


const SQLQueryForm = (props) => {
    const queryHandler = () => {
        const data = createJsonData(['id', 'query'], [
            props.fileId, props.query
        ]);
        props.onSettingsApply('/api/execute-query/', data);
    };
    return (
        <div className={"col-md-4 mt-3"}>
            <MDBInputGroup
                material
                containerClassName="mb-3 mt-0"
                prepend={
                    <MDBBtn className="m-0 px-3 py-2 z-depth-0" onClick={queryHandler}>
                        <MDBIcon icon={"play"} />
                    </MDBBtn>
                }
                inputs={
                    <>
                        <MDBInput noTag  hint={"SQL"} type="text" id={"query"} onChange={props.onColumnsAndRowsValueChange} />
                    </>
                }
        />
        </div>
    )
};

const mapStateToProps = state => {
    return {
        fileId: state.fileUpload.id,
        query: state.parameters.query
    }
};

const mapDispatchToProps = dispatch => {
    return {
        onSettingsApply: (url, data) => dispatch(applySettings(url, data)),
        onColumnsAndRowsValueChange: event => dispatch(columnsAndRowsValueChangeHandler(event))
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(SQLQueryForm);