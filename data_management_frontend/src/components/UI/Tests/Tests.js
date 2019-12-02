import React, {useState, useEffect, useRef} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import axios from "./../../../instanceAxios";
import { mapDataColumns, createJsonData } from '../../../utility/utility';
import { connect } from "react-redux";
import ResultsTable from "../ResultsTable/ResultsTable";
import Alert from "../Alert/Alert";
import Spinner from "../Spinner/Spinner";


const useStyles = makeStyles(theme => ({
    formControl: {
      margin: theme.spacing(1),
      minWidth: 120,
    },
    selectEmpty: {
      marginTop: theme.spacing(2),
    },
}));

const Tests = ({ fileId, data }) => {
    const classes = useStyles();
    const [result, setResult] = useState('');
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(false);
    const [test, setTest] = useState('normtest');
    const [xValue, setXValue] = useState(mapDataColumns(data.columns)[0]);
    const [yValue, setYValue] = useState(mapDataColumns(data.columns)[0]);

    useEffect(() => {
        startTest();
    }, [test, xValue, yValue]);

    const startTest = () => {
        const data = createJsonData(['id', 'x', 'y', 'test'], [fileId, xValue, yValue, test]);
        setLoading(true);
        axios.post('/api/stats/', data)
        .then(response => {
            setResult(response.data.result);
            setError(response.data.error);
            setLoading(false);
        }).catch(error => {
            setError(error.message);
            setLoading(false);
        })

    };

    const handleChange = event => {
        setTest(event.target.value);
    };

    const handleXChange = event => {
        setXValue(event.target.value);
    };

    const handleYChange = event => {
        setYValue(event.target.value);
    }
    let content = null;
    if (result !== '') {
        content = <ResultsTable test={test} result={result} />
    }
    else if (error) {
        content = <Alert>{error}</Alert>
    }
    
    return (
        <div className="row d-flex justify-content-center">
            <div className="col-md-4 d-flex justify-content-center">
                <FormControl className={classes.formControl}>
                    <InputLabel id="plot-type">Tests Statistiques</InputLabel>
                    <Select
                        labelId="plot-type"
                        id="select-plot"
                        value={test}
                        onChange={handleChange}
                        >
                        <MenuItem value={'normtest'}>Test de Normalité</MenuItem>
                        <MenuItem value={'skewtest'}>Test de symétrie</MenuItem>
                        <MenuItem value={'cumfreq'}>Test de fréquence</MenuItem>
                        <MenuItem value={'correlation'}>Test de correlation</MenuItem>
                        <MenuItem value={'t-test'}>Test de Student ou t-test</MenuItem>
                        <MenuItem value={'anova'}>Test ANOVA</MenuItem>
                        <MenuItem value={'chisquare'}>Test de Chi2</MenuItem>
                        <MenuItem value={'fisher_exact'}>Test exact de Fisher</MenuItem>
                        <MenuItem value={'wilcoxon'}>Test de WILCOXON</MenuItem>
                        <MenuItem value={'zscore'}>Test du Z-score</MenuItem>
                    </Select>
                </FormControl>
            </div>
            <div className="col-md-4 d-flex justify-content-center">
                <FormControl className={classes.formControl}>
                    <InputLabel id="xValue">Set X value</InputLabel>
                    <Select
                        labelId="xValue"
                        id="select-value"
                        value={xValue}
                        onChange={handleXChange}
                        >
                        {data.columns.map(column => <MenuItem value={column.label}>{column.label}</MenuItem>)}
                    </Select>
                </FormControl>
            </div>
            <div className="col-md-4 d-flex justify-content-center">
                <FormControl className={classes.formControl}>
                    <InputLabel id="xValue">Set Y value</InputLabel>
                    <Select
                        labelId="xValue"
                        id="select-value"
                        value={yValue}
                        onChange={handleYChange}>
                        {data.columns.map(column => <MenuItem value={column.label}>{column.label}</MenuItem>)}
                    </Select>
                </FormControl>
            </div>
            <div className="col-md-12 d-flex justify-content-center">
                {
                    loading ? <Spinner /> : content
                }
            </div>
        </div>
    )
};

export const mapStateToProps = state => {
    return {
        fileId: state.fileUpload.id,
        data: state.fileUpload.file_data,
    }
}


export default connect(mapStateToProps)(Tests);