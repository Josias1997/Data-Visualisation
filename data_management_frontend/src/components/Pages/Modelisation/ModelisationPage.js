import React, { useEffect, useState } from 'react';
import { MDBContainer, MDBBtn, MDBIcon } from 'mdbreact';
import DataTable from '../../UI/DataTable/DataTable';
import { connect } from "react-redux";
import AlignCenter from "../../UI/AlignCenter/AlignCenter";
import Form from "../../UI/Form/Form";
import Spinner from '../../UI/Spinner/Spinner';
import Table from '../../UI/Table/Table';
import TableWithNumericValues from '../../UI/TableWithNumericValues/TableWithNumericValues';
import axios from '../../../instanceAxios';
import { createJsonData } from '../../../utility/utility';
import MaterialTable from 'material-table';
import { makeStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import { splitDataSet, normalize } from "../../../store/actions/";
import { options } from '../../../utility/settings';

const useStyles = makeStyles(theme => ({
    formControl: {
      margin: theme.spacing(1),
      minWidth: 300,
    },
    selectEmpty: {
      marginTop: theme.spacing(2),
    },
}));

const ModelisationPage = props => {
    const classes = useStyles();
    const [normalizer, setNormalizer] = useState('std_scaler');

    useEffect(() => {
        if(props.id !== undefined) {
            splitDataSet();
        }
    }, [props.id]);


    const splitDataSet = () => {
        const data = createJsonData(['id'], [props.id]);
        props.onSplitDataSet(data);
    };
    const handleChange = (event) => {
        setNormalizer(event.target.value);
    };

    const normalize = () => {
        const data = createJsonData(['id', 'normalizer'], [props.id, normalizer]);
        props.onNormalize(data);
    }

    return (
        <MDBContainer>
            {
                props.id !== undefined ? <>
                <div className="container justify-content-center mt-5 mb-3">
                    <FormControl className={classes.formControl}>
                        <InputLabel id="function">Normaliser les données</InputLabel>
                        <Select
                            labelId="function"
                            id="function"
                            value={normalizer}
                            onChange={handleChange}
                            >
                            <MenuItem value='std_scaler'>StandardScaler</MenuItem>
                            <MenuItem value='min_max_scaler'>MinMaxScaler</MenuItem>
                            <MenuItem value='robust_scaler'>RobustScaler</MenuItem>
                            <MenuItem value='normalizer'>Normalizer</MenuItem>
                        </Select>
                    </FormControl>
                    <MDBBtn onClick={normalize}> <MDBIcon className="mr-2" icon={"play"} /> Go</MDBBtn>
                </div>
                {
                    (!props.processing ? <> 
                        <div className="container justify-content-center mt-5 mb-3">
                        {
                            props.normalized ? <TableWithNumericValues data={props.normalizedTrainingSet} /> : null
                        }
                </div>
                <div className="container justify-content-center mt-5 mb-3">
                    <MaterialTable
                    title={"Training set"}
                    columns={props.trainingSet.columns}
                    data={props.trainingSet.rows}
                    options={options}
                />
                </div>
                <div className="container justify-content-center mt-5 mb-3">
                    <MaterialTable
                    title={"Test set"}
                    columns={props.testSet.columns}
                    data={props.testSet.rows}
                    options={options}
                 />
                 </div></> : <Spinner />)
                }

                </> : <AlignCenter style={{
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
        trainingSet: state.modelisation.trainingSet,
        testSet: state.modelisation.testSet,
        processing: state.modelisation.processing,
        error: state.modelisation.error,
        normalized: state.modelisation.normalized,
        normalizedTrainingSet: state.modelisation.normalizedTrainingSet
    }
}

const mapDispatchToProps = dispatch => {
    return {
        onSplitDataSet: (data) => dispatch(splitDataSet(data)),
        onNormalize: (data) => dispatch(normalize(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(ModelisationPage);