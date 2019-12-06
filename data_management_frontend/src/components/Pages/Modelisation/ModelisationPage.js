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
    const [trainingSet, setTrainingSet] = useState({});
    const [testSet, setTestSet] = useState({});
    const [processing, setProcessing] = useState(false);
    const [error, setError] = useState(false);
    const [normalizer, setNormalizer] = useState('std_scaler');
    const [normalizedTrainingSet, setNormalizedTrainingSet] = useState([]);
    const [normalized, setNormalized] = useState(false);
    const options = {
        filterType: "dropdown",
        responsive: "scroll"
    };

    useEffect(() => {
        if(props.id !== undefined) {
            splitDataSet();
        }
    }, [props.id]);


    const splitDataSet = () => {
        const data = createJsonData(['id'], [props.id]);
        setProcessing(true);
        axios.post('/api/split-data-set/', data)
        .then(({data}) => {
            setTrainingSet(data.training_set);
            setTestSet(data.test_set);
            setError(data.error);
            setProcessing(false);
        }).catch(error => {
            setError(error.message);
            setProcessing(false);
        });
    };
    const handleChange = (event) => {
        setNormalizer(event.target.value);
    };

    const normalize = () => {
        const data = createJsonData(['id', 'normalizer'], [props.id, normalizer]);
        setProcessing(true);
        axios.post('/api/preprocessing/', data)
        .then(({data}) => {
            console.log(data);
            setNormalizedTrainingSet(data.normalized_training_set);
            setError(data.error);
            setProcessing(false);
            setNormalized(true);
        }).catch(error => {
            setError(error.message);
            setProcessing(false);
        })
    }

    return (
        <MDBContainer>
            {
                props.id !== undefined ? (!processing ? <>
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
                    <MDBBtn onClick={normalize}> <MDBIcon icon={"play"} /> Go</MDBBtn>
                </div>
                <div className="container justify-content-center mt-5 mb-3">
                {
                    normalized ? <TableWithNumericValues data={normalizedTrainingSet} /> : null
                }
                </div>
                <div className="container justify-content-center mt-5 mb-3">
                    <MaterialTable
                    title={"Training set"}
                    columns={trainingSet.columns}
                    data={trainingSet.rows}
                    options={options}
                />
                </div>
                <div className="container justify-content-center mt-5 mb-3">
                    <MaterialTable
                    title={"Test set"}
                    columns={testSet.columns}
                    data={testSet.rows}
                    options={options}
                 />
                 </div>
                </> : <Spinner />) : <AlignCenter style={{
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