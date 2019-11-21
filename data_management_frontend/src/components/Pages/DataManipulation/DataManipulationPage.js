import React, {Fragment} from 'react';
import Form from "../../UI/Form/Form";
import Settings from "../../UI/Settings/Settings";
import DataTable from "../../UI/DataTable/DataTable";
import {MDBCol, MDBContainer} from "mdbreact";
import AlignCenter from "../../UI/AlignCenter/AlignCenter";
import {connect} from 'react-redux';
import Spinner from '../../UI/Spinner/Spinner';

const DataManipulationPage = props => {
    return (
        <MDBContainer>
            {
                props.fileId ? <Fragment>
                    <AlignCenter>
                        <MDBCol col={12}>
                            <Settings page="data-manipulation"/>
                        </MDBCol>
                    </AlignCenter>
                    <MDBCol col={12}>
                        <hr style={{
                            border: '2px solid #ccc',
                        }}/>
                    </MDBCol>
                    <AlignCenter>
                        <MDBCol col={12}>
                            <DataTable/>
                        </MDBCol>
                    </AlignCenter>
                </Fragment> : <AlignCenter style={{
                    marginTop: '15%'
                }}>
                    {
                        !props.loading ? <Form /> : <Spinner />
                    }
                </AlignCenter>
            }

        </MDBContainer>
    );
};

const mapStateToProps = state => {
    return {
        fileId: state.fileUpload.id,
        loading: state.fileUpload.loading
    }
};

export default connect(mapStateToProps)(DataManipulationPage);