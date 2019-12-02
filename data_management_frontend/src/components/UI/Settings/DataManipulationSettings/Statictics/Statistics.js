import React from 'react';
import { connect } from 'react-redux';
import {applySettings} from "../../../../../store/actions";
import {createJsonData} from "../../../../../utility/utility";
import { MDBIcon, MDBBtn } from "mdbreact";

const Statistics = ({fileId, onDescribeHandler}) => {

    const describe = () => {
        const data = createJsonData(['id'], [fileId]);
        onDescribeHandler('/api/describe/', data);
    };

    return (<MDBBtn color="default" onClick={describe}>
                <MDBIcon icon="chart-line" className="mr-1"/> Stats
    </MDBBtn>);
};

const mapStateToProps = state => {
    return {
        fileId: state.fileUpload.id,
    }
};

const mapDispatchToProps = dispatch => {
    return {
        onDescribeHandler: (url, data) => dispatch(applySettings(url, data))
    }
};
export default connect(mapStateToProps, mapDispatchToProps)(Statistics);