import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Input from '@material-ui/core/Input';
import Select from '@material-ui/core/Select';
import Chips from "react-chips";
import axios from "../../../instanceAxios";
import { connect } from "react-redux";
import { createJsonData, mapDataColumns } from '../../../utility/utility';
import { addPlot } from '../../../store/actions';
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

const Plot = ({ fileId, data, onAddPlot }) => {
    const classes = useStyles();
    const [columns, setColumns] = useState(mapDataColumns(data.columns));
    const [plotType, setPlotType] = useState('bar');
    const [plotPath, setPlotPath] = useState('');
    const [xValue, setXValue] = useState(mapDataColumns(data.columns)[0])
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(false);

    const [columnsColors, setColumnsColors] = useState(data.columns.map(column => {
        return {
            id: column.field,
            value: '#000000',
        }
    }));

    useEffect(() => {
        plot();
    }, [columns, plotType, xValue, columnsColors]);

    const plot = () => {
        let colors = columnsColors.filter(column => columns.indexOf(column.id) !== -1);
        colors = colors.map(color => color.value);
        const data = createJsonData(['id', 'x', 'columns', 'kind', 'y_colors'], 
            [fileId, xValue, columns, plotType, colors]);
        setLoading(true);
        axios.post('/api/plot/', data)
        .then(({data}) => {
            if (data.plot !== '') {
                onAddPlot(data.plot);
            }
            setPlotPath(data.plot);
            setLoading(false);
        }).catch(error => {
            setError(error.message);
            setLoading(false);
        })
    };
    const handleColumnsColorChange = event => {
        setColumnsColors(current => {
            const newArray = [...current];
            let index = newArray.findIndex(color => color.id === event.target.id);
            if (index === -1) {
                return;
            } else {
               newArray[index].value = event.target.value;
            }
            console.log(newArray);
            return newArray; 
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
        content = <img src={plotPath}/>
    } else if (error) {
        content = <Alert>
            {error}
        </Alert>
    }
    return (
        <div className="row d-flex justify-content-center">
            <label>Set Y Value</label>
            <div className="col-md-12">
                <Chips
                    value={columns}
                    onChange={onChange}
                    suggestions={[...mapDataColumns(data.columns)]}

                />
            </div>
            <div className="col-md-12">
                <div className="row">
                {
                    columns.map(column => <Input style={{
                        width: column.length >= 9 ? '100px' : '50px',
                        marginLeft: '14px'
                    }} key={column} onChange={handleColumnsColorChange} type="color" id={column} />)
                }
                </div>
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
                            <MenuItem value={'barh'}>Bar Chart Horizontal</MenuItem>
                            <MenuItem value={'bar'}>Bar Chart Vertical</MenuItem>
                            <MenuItem value={'hist'}>Bar Histogram</MenuItem>
                            <MenuItem value={'area'}>Stacked Area</MenuItem>
                            <MenuItem value={'box'}>Box plot</MenuItem>
                            <MenuItem value={'pie'}>Pie Chart</MenuItem>
                            <MenuItem value={'line'}>Line Chart</MenuItem>
                            <MenuItem value={'scatter'}>Scatter</MenuItem>
                            <MenuItem value={'kde'}>KDE</MenuItem>
                            <MenuItem value={'correlogram'}>Correlogram</MenuItem>
                            <MenuItem value={'density-plot'}>Density Plot</MenuItem>
                            <MenuItem value={'treemap'}>Treemap</MenuItem>
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
                                data.columns.map(column => <MenuItem value={column.field}>{column.field}</MenuItem>)
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

const mapDispatchToProps = dispatch => {
    return {
        onAddPlot: path => dispatch(addPlot(path))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Plot);