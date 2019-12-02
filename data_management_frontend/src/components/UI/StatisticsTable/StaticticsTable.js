import React from "react";
import Spinner from "../Spinner/Spinner";
import { connect } from 'react-redux';
import Pivot from '../../../webdatarocks.react';


const StatisticsTable = ({loading, path}) => {
    return (
        <div className="container justify-content-center mt-5 mb-3">
            {
                !loading ? <Pivot toolbar={true} report={{
                    "dataSource": {
                        "dataSourceType": "csv",
                        "filename": path
                    }}
                }/>: <Spinner/>
            }
        </div>
    );
};

const mapStateToProps = state => {
    return {
        loading: state.fileUpload.loading,
        path: state.fileUpload.path
    }
};

export default connect(mapStateToProps)(StatisticsTable);