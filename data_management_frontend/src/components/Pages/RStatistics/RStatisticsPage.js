import React, {Fragment} from 'react';
import { MDBContainer, MDBCol } from 'mdbreact';
import { connect } from "react-redux";
import AlignCenter from "../../UI/AlignCenter/AlignCenter";
import Form from "../../UI/Form/Form";
import Settings from "../../UI/Settings/Settings";
import StaticticsTable from '../../UI/StatisticsTable/StaticticsTable';
import Plot from '../../UI/Plot/Plot';
import Tests from '../../UI/Tests/Tests';
import Spinner from '../../UI/Spinner/Spinner';

const RStatisticsPage = props => {
    let content = null;
    if (props.openTable) {
        content = <StaticticsTable />
    }
    else if (props.openPlot) {
        content = <Plot />
    }
    else if (props.openTests) {
        content = <Tests />
    }
    return (
        <MDBContainer>
            {
                props.id !== undefined ? <Fragment>
                <AlignCenter>
                    <MDBCol col={12}>
                        <Settings page="r-statistics" />
                    </MDBCol>
                </AlignCenter>
                <MDBCol col={12}>
                    <hr style={{
                        border: '2px solid #ccc',
                    }}/>
                </MDBCol>
                <AlignCenter>
                    <MDBCol col={12}>
                        {content}
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
        id: state.fileUpload.id,
        openPlot: state.statistics.openPlot,
        openTable: state.statistics.openTable,
        openTests: state.statistics.openTests,
        loading: state.fileUpload.loading,
    }
};

export default connect(mapStateToProps)(RStatisticsPage);