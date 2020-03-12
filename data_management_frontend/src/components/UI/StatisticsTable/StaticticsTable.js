import React, {useState, useEffect, useRef} from "react";
import { MDBBtn, MDBIcon } from 'mdbreact';
import Spinner from "../Spinner/Spinner";
import { connect } from 'react-redux';
import Table from '../Table/Table';
import { makeStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import { getFunctionsList, createJsonData } from "../../../utility/utility";
import axios from "../../../instanceAxios";
import * as WebDataRocksReact from './../../../webdatarocks.react.js';


const useStyles = makeStyles(theme => ({
    formControl: {
      margin: theme.spacing(1),
      minWidth: 120,
    },
    selectEmpty: {
      marginTop: theme.spacing(2),
    },
}));


const StatisticsTable = ({loading, path, data, fileId}) => {
    const classes = useStyles();
    const [functionName, setFunctionName] = useState('');
    const [xValue, setXValue] = useState('');
    const [computing, setComputing] = useState(false);
    const [error, setError] = useState(false);
    const [resultData, setResultData] = useState(data);

    useEffect(() => {
        if (xValue !== '' && functionName !== '') {
            callFunction();
        }
    }, [functionName, xValue]);

    const handleXChange = (event) => {
        setXValue(event.target.value);
    };

    const handleChange = (event) => {
        setFunctionName(event.target.value);
    };

    const resetData = () => {
        setResultData(current => data);
    };

    const callFunction = () => {
        const data = createJsonData(['id', 'function', 'x'], [fileId, functionName, xValue]);
        setComputing(true);
        axios.post('/api/math-functions/', data)
        .then(response => {
            setResultData(response.data);
            setError(false);
            setComputing(false);
        }).catch(error => {
            console.log(error);
            setComputing(false);
        })
    };
    return (
        <div className="container d-flex justify-content-center mt-5 mb-3">
            {
                !loading ?<> 
                    <div className="col-md-8">
                    {
                        !computing ? <WebDataRocksReact.Pivot toolbar={true} report={{
                            dataSource: {
                                data: resultData.rows
                            }
                        }}/> : <Spinner />
                    }
                    </div>
                    <div className="col-md-1">
                        <hr style={{
                            width: '0.5px',
                            height: '100%',
                            border: '1px solid black',
                    }} />
                    </div>
                    <div className="col-md-3">
                        <FormControl className={classes.formControl}>
                            <InputLabel id="function">Fonctions</InputLabel>
                            <Select
                                labelId="function"
                                id="function"
                                value={functionName}
                                onChange={handleChange}
                                >
                                {
                                    getFunctionsList().map(name => <MenuItem key={name} value={name}>{name}</MenuItem>)
                                }
                            </Select>
                        </FormControl>
                    <FormControl className={classes.formControl}>
                        <InputLabel id="xValue">Valeur de X</InputLabel>
                        <Select
                            labelId="xValue"
                            id="select-value"
                            value={xValue}
                            onChange={handleXChange}
                            >
                            {
                                data.columns.map(column => <MenuItem key={column} value={column.field}>{column.field}</MenuItem>)
                            }
                        </Select>
                    </FormControl>
                    <MDBBtn color={"danger"} onClick={resetData}><MDBIcon icon={"angle-left"} className="mr-2"></MDBIcon>Precedent</MDBBtn>
                    </div>
                </>
                : <Spinner/>
            }
        </div>
    );
};

const mapStateToProps = state => {
    return {
        loading: state.fileUpload.loading,
        path: state.fileUpload.path,
        data: state.fileUpload.file_data,
        fileId: state.fileUpload.id,
    }
};


export default connect(mapStateToProps)(StatisticsTable);