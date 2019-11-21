import React from "react";
import Spinner from "../Spinner/Spinner";
import MUIDataTable from 'mui-datatables';
import { connect } from 'react-redux';


const DataTable = ({loading, data, name}) => {
    const options = {
        filterType: "dropdown",
        responsive: "scroll"
    };
    return (
        <div className="container justify-content-center mt-5 mb-3">
            {
                !loading ? <MUIDataTable
                    title={"Données du fichier " + name}
                    columns={data.columns}
                    data={data.rows}
                    options={options}
                /> : <Spinner/>
            }
        </div>
    );
};

const mapStateToProps = state => {
    return {
        name: state.fileUpload.name,
        data: state.fileUpload.file_data,
        loading: state.fileUpload.loading
    }
};

export default connect(mapStateToProps)(DataTable);