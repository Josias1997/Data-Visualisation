import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import Chips from "react-chips";
import axios from "../../../instanceAxios";
import { connect } from "react-redux";
import { createJsonData, mapDataColumns } from '../../../utility/utility';
import Spinner from '../Spinner/Spinner';
import Alert from '../Alert/Alert';

const useStyles = makeStyles(theme => ({
    formControl: {
      margin: theme.spacing(1),
      minWidth: 120,
    },
    selectEmpty: {
      marginTop: theme.spacing(2),
    },
}));

const Plot = ({ fileId, data }) => {
    const classes = useStyles();
    const [columns, setColumns] = useState(mapDataColumns(data.columns));
    const [plotType, setPlotType] = useState('bar');
    const [plotPath, setPlotPath] = useState('');
    const [xValue, setXValue] = useState(mapDataColumns(data.columns)[0])
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        plot();
    }, [columns, plotType, xValue]);

    const plot = () => {
        const data = createJsonData(['id', 'x', 'columns', 'kind'], [fileId, xValue, columns, plotType]);
        setLoading(true);
        axios.post('/api/plot/', data)
        .then(response => {
            setPlotPath(response.data.plot);
            setError(response.data.error);
            setLoading(false);
        }).catch(error => {
            setError(error.message);
            setLoading(false);
        })
    };
    const handleChange = event => {
        setPlotType(event.target.value);
    };
     const onChange = chips => {
        setColumns(chips);
    };

    const handleXChange = event => {
        setXValue(event.target.value);
    };

    let content = null;
    if (plotPath !== '') {
        content = <img src={plotPath} style={{
            width: '700px'
        }} />
    } else if (error) {
        content = <Alert>
            {error}
        </Alert>
    }
    return (
        <div className="row d-flex justify-content-center">
            <div className="col-md-12">
                <Chips
                    value={columns}
                    onChange={onChange}
                    suggestions={[...mapDataColumns(data.columns)]}

                />
            </div>
            <div className="row d-flex justify-content-center">
                <div className="col-md-6 d-flex justify-content-center">
                    <FormControl className={classes.formControl}>
                        <InputLabel id="plot-type">Plot Type</InputLabel>
                        <Select
                            labelId="plot-type"
                            id="select-plot"
                            value={plotType}
                            onChange={handleChange}
                            >
                            <MenuItem value={'bar'}>Bar</MenuItem>
                            <MenuItem value={'hist'}>Histogram</MenuItem>
                            <MenuItem value={'box'}>Box</MenuItem>
                            <MenuItem value={'kde'}>KDE</MenuItem>
                            <MenuItem value={'area'}>Area</MenuItem>
                            <MenuItem value={'scatter'}>Scatter</MenuItem>
                            <MenuItem value={'pie'}>Pie</MenuItem>
                        </Select>
                    </FormControl>
                </div>
                <div className="col-md-6 d-flex justify-content-center">
                    <FormControl className={classes.formControl}>
                        <InputLabel id="xValue">Set X value</InputLabel>
                        <Select
                            labelId="xValue"
                            id="select-value"
                            value={xValue}
                            onChange={handleXChange}
                            >
                            {
                                data.columns.map(column => <MenuItem value={column.label}>{column.label}</MenuItem>)
                            }
                        </Select>
                    </FormControl>
                </div>
            </div>
            <div className="col-md-12 d-flex justify-content-center">
                {
                    !loading ? content : <Spinner />
                }
            </div>
        </div>
    )
};

const mapStateToProps = state => {
    return {
        fileId: state.fileUpload.id,
        data: state.fileUpload.file_data,
    }
};

export default connect(mapStateToProps)(Plot);