import React from "react";
import Spinner from "../Spinner/Spinner";
import MaterialTable from "material-table";
import { connect } from 'react-redux';
import { updateDataSuccess } from '../../../store/actions';
import { options, localization } from '../../../utility/settings';

const DataTable = ({loading, data, name, path, updateData}) => {

    const addRow = (newData) => {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                let data_copy = JSON.parse(JSON.stringify(data));
                data_copy.rows.push(newData);
                updateData(data_copy);
                resolve();
            }, 1000);
        });
    };

    const updateRow = (newData, oldData) => {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                let data_copy = JSON.parse(JSON.stringify(data));
                let index = data_copy.rows.findIndex(row => JSON.stringify(row) === JSON.stringify(oldData));
                data_copy.rows[index] = newData;
                updateData(data_copy);
                resolve();
            }, 1000);
        });
    };

    const deleteRow = (oldData) => {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                let data_copy = JSON.parse(JSON.stringify(data));
                let index = data_copy.rows.findIndex(row => JSON.stringify(row) === JSON.stringify(oldData));
                data_copy.rows.splice(index, 1);
                updateData(data_copy);
                resolve();
            }, 1000);
        });
    };
    return (
        <div className="container justify-content-center mt-5 mb-3">
            {
                !loading ? <MaterialTable
                title={"Données du fichier " + name}
                columns={data.columns}
                data={data.rows}
                options={options}
                editable={{
                    onRowAdd: newData => addRow(newData),
                    onRowUpdate: (newData, oldData) => updateRow(newData, oldData),
                    onRowDelete: oldData => deleteRow(oldData)
                }}
                localization={localization}
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

const mapDispatchToProps = dispatch => {
    return {
        updateData: data => dispatch(updateDataSuccess(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(DataTable);