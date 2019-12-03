import React from 'react';
import { MDBContainer } from 'mdbreact';
import DataTable from '../../UI/DataTable/DataTable';
import { connect } from "react-redux";
import AlignCenter from "../../UI/AlignCenter/AlignCenter";
import Form from "../../UI/Form/Form";
import Spinner from '../../UI/Spinner/Spinner';

const ModelisationPage = props => {
    return (
        <MDBContainer>
            {
                props.id !== undefined ? <DataTable /> : <AlignCenter style={{
                    marginTop: '15%'
                }}>
                    {
                        !props.loading ? <Form/> : <Spinner />
                    }
                </AlignCenter>
            }
        </MDBContainer>
    )
};

const mapStateToProps = state => {
    return {
        id: state.fileUpload.id,
        loading: state.fileUpload.loading,
    }
}

export default connect(mapStateToProps)(ModelisationPage);