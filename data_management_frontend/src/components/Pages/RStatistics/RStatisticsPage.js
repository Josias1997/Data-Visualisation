import React, {Fragment} from 'react';
import { MDBContainer, MDBCol } from 'mdbreact';
import DataTable from '../../UI/DataTable/DataTable';
import { connect } from "react-redux";
import AlignCenter from "../../UI/AlignCenter/AlignCenter";
import Form from "../../UI/Form/Form";
import Settings from "../../UI/Settings/Settings";

const RStatisticsPage = props => {
    return (
        <MDBContainer>
            {
                props.id !== undefined ? <Fragment>
                <AlignCenter>
                    <MDBCol col={12}>
                        <Settings page="r-statistics"/>
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
                    <Form/>
                </AlignCenter>
            }
           
        </MDBContainer>
    );
};
const mapStateToProps = state => {
    return {
        id: state.fileUpload.id,
    }
}

export default connect(mapStateToProps)(RStatisticsPage);