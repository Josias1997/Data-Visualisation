import React from "react";
import Spinner from "../Spinner/Spinner";
import { connect } from 'react-redux';
import Pivot from '../../../webdatarocks.react';
import Table from '../Table/Table';


const StatisticsTable = ({loading, path, data}) => {
    return (
        <div className="container justify-content-center mt-5 mb-3">
            {
                !loading ? <Table columns={data.columns} rows={data.rows}/>: <Spinner/>
            }
        </div>
    );
};

const mapStateToProps = state => {
    return {
        loading: state.fileUpload.loading,
        path: state.fileUpload.path,
        data: state.fileUpload.file_data,
    }
};

export default connect(mapStateToProps)(StatisticsTable);