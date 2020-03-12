import React, {Fragment} from 'react';
import Form from "../../UI/Form/Form";
import Settings from "../../UI/Settings/Settings";
import DataTable from "../../UI/DataTable/DataTable";
import {MDBCol, MDBContainer} from "mdbreact";
import AlignCenter from "../../UI/AlignCenter/AlignCenter";
import {connect} from 'react-redux';
import Spinner from '../../UI/Spinner/Spinner';

const DataManipulationPage = props => {
    const style = {
        width: '1100px'
    };

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
                        {
                            props.seabornPlot === '' ? <DataTable/> : <> <img src={props.seabornPlot} alt="Seaborn Plot" style={style} /> 
                            <img src={props.barPlot} alt="Bar Plot" style={style} />
                            <img src={props.heatmapPlot} alt="Heatmap Plot" style={style} />
                            <img src={props.matrixPlot} alt="Matrix Plot" style={style}/>
                            </>
                        }
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
        loading: state.fileUpload.loading,
        seabornPlot: state.fileUpload.seabornPlot,
        barPlot: state.fileUpload.barPlot,
        heatmapPlot: state.fileUpload.heatmapPlot,
        matrixPlot: state.fileUpload.matrixPlot,
    }
};

export default connect(mapStateToProps)(DataManipulationPage);