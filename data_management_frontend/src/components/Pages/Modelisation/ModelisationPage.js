import React from 'react';
import { MDBContainer } from 'mdbreact';
import DataTable from '../../UI/DataTable/DataTable';
import { connect } from "react-redux";
import AlignCenter from "../../UI/AlignCenter/AlignCenter";
import Form from "../../UI/Form/Form";

const ModelisationPage = props => {
    return (
        <MDBContainer>
            {
                props.id !== undefined ? <DataTable /> : <AlignCenter style={{
                    marginTop: '15%'
                }}>
                    <Form/>
                </AlignCenter>
            }
        </MDBContainer>
    )
};

const mapStateToProps = state => {
    return {
        id: state.fileUpload.id,
    }
}

export default connect(mapStateToProps)(ModelisationPage);