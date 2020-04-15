import React, { useState, useEffect, useRef } from "react";
import { connect } from "react-redux";
import { options } from '../../../utility/settings';
import { createJsonData, convertToPDF, convertToJPEG, convertToPNG } from '../../../utility/utility';
import MaterialTable from "material-table";
import AlignCenter from "../../UI/AlignCenter/AlignCenter";
import Matrix from "../../UI/Matrix/Matrix";
import Form from "../../UI/Form/Form";
import Spinner from '../../UI/Spinner/Spinner';
import Settings from "../../UI/Settings/Settings";
import { MDBContainer, MDBCol } from "mdbreact";
import { fit, predict, splitDataSet } from '../../../store/actions/';
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


const MachineLearning = (props) => {
    const classes = useStyles();
    const [yValue, setYValue] = useState('');
    const [xValue, setXValue] = useState('');
    const [algorithm, setAlgorithm] = useState('');
    const toBeDownloaded = useRef(null);

    const handleAgorithmChange = (event) => {
        setAlgorithm(event.target.value);
    };

    const handleXChange = (event) => {
        setXValue(event.target.value);
    };

    const handleYChange = (event) => {
        setYValue(event.target.value);
    };

    const predict = () => {
        const data = createJsonData(['id', 'x', 'y', 'algorithm'], [props.fileId, xValue, yValue, algorithm]);
        props.onPredict(data, 'deep_learning');
    };

    const split = () => {
        if (props.fileId !== undefined) {
            const data = createJsonData(['id'], [props.fileId]);
            props.onSplitDataSet(data);
        }
    };

    const print = (type) => {
        if( type === "jpeg") {
            convertToJPEG(algorithm, toBeDownloaded.current);
        }
        else if (type === "pdf") {
            convertToPDF(algorithm, toBeDownloaded.current);
        }
        else if (type === "png") {
            convertToPNG(algorithm, toBeDownloaded.current);
        }
    };

	return (
		<MDBContainer>
			{
				props.fileId ? <>
				 <AlignCenter>
                    <MDBCol col={12}>
                        <Settings page="machine-learning" onPrint={print} onPredict={predict} onSplit={split}/>
                    </MDBCol>
                </AlignCenter>
                <MDBCol col={12}>
                    <hr style={{
                        border: '2px solid #ccc',
                    }}/>
                </MDBCol>
                <div className="container justify-content-center mt-5 mb-3">
                {
                    (algorithm === 'artificial-neural-network' || algorithm === 'convolutional-neural-network'
                    || algorithm === 'recurrent-neural-network') 
                    ? null : <>
                        <FormControl className={classes.formControl}>
                        <InputLabel id="independantVariable">X</InputLabel>
                        <Select
                            labelId="independantVariable"
                            id="independantVariable"
                            value={xValue}
                            onChange={handleXChange}
                            >
                            {
                                props.data.columns.map(column => <MenuItem key={column.field} value={column.field}>{column.field}</MenuItem>)
                            }
                        </Select>
                    </FormControl>
                    <FormControl className={classes.formControl}>
                        <InputLabel id="dependantVariable">Y</InputLabel>
                        <Select
                            labelId="dependantVariable"
                            id="dependantVariable"
                            value={yValue}
                            onChange={handleYChange}
                            >
                            {
                                props.data.columns.map(column => <MenuItem key={column.field} value={column.field}>{column.field}</MenuItem>)
                            }
                        </Select>
                    </FormControl>
                    </>
                }
                    <FormControl className={classes.formControl}>
                        <InputLabel id="dependantVariable">Algorithm</InputLabel>
                        <Select
                            labelId="dependantVariable"
                            id="dependantVariable"
                            value={algorithm}
                            onChange={handleAgorithmChange}
                            >
                            <MenuItem value={'artificial-neural-network'}>Artificial Neural Network</MenuItem>
                            <MenuItem value={'convolutional-neural-network'}>Convolutional Neural Network</MenuItem>
                            <MenuItem value={'recurrent-neural-network'}>Recurrent Neural Network</MenuItem>
                        </Select>
                    </FormControl>
                </div>
                {
                (props.processing ? <Spinner /> : <>
                     <div className="container col-md-12 justify-content-center mt-5 mb-3">

                        <div className="container" ref={toBeDownloaded} id="to-be-downloaded" style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}>
                        {
                            algorithm === 'recurrent-neural-network' ? <>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.stockPricePlot} alt="Stock Price"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.lstmPlot} alt="Stock Price"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.gruPlot} alt="GRU"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.sequencePlot} alt="Sequence"/>
                                </div></> : null
                        }
                        {
                                algorithm === 'artificial-neural-network' ? <>
                                <div className=" mt-2 container justify-content-center">
                                    {props.confusionMatrix !== undefined ? <Matrix matrix={props.confusionMatrix} /> : null}
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.matrixPlot} alt="Confusion Matrix"/>
                                </div>
                            </> : null
                        }
                        {
                            algorithm === 'convolutional-neural-network' ? <>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.samples1} alt="Stock Price"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.samples2} alt="GRU"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.examples} alt="Sequence"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.model} alt="Stock Price"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.matrixPlot} alt="GRU"/>
                                </div>
                            </> : null
                        }
                        </div>
                </div>
                {
                    props.predicted ? null : (props.splitProcessing ? <Spinner /> : <> 
                    <div className="container justify-content-center mt-5 mb-3">
                    {
                        Object.entries(props.trainingSet).length !== 0 ? <MaterialTable
                        title={"Training set"}
                        columns={props.trainingSet.columns}
                        data={props.trainingSet.rows}
                        options={options}
                    /> : null
                    }
                    </div>
                    <div className="container justify-content-center mt-5 mb-3">
                    {
                        Object.entries(props.testSet).length !== 0 ? <MaterialTable
                        title={"Test set"}
                        columns={props.testSet.columns}
                        data={props.testSet.rows}
                        options={options}
                        /> : null
                    }
                    </div>
                </>)
                }
        </>)
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
        data: state.fileUpload.file_data,
		fileId: state.fileUpload.id,
		loading: state.fileUpload.loading,
		trainingSet: state.modelisation.trainingSet,
		testSet: state.modelisation.testSet,
        processing: state.deep_learning.loading,
        predicted: state.deep_learning.predicted,
        splitProcessing: state.modelisation.processing,
        confusionMatrix: state.deep_learning.confusionMatrix,
        stockPricePlot: state.deep_learning.stockPricePlot,
        lstmPlot: state.deep_learning.lstmPlot,
        gruPlot: state.deep_learning.gruPlot,
        sequencePlot: state.deep_learning.sequencePlot,
        samples1: state.deep_learning.samples1,
        samples2: state.deep_learning.samples2,
        examples: state.deep_learning.examples,
        model: state.deep_learning.model,
        matrixPlot: state.deep_learning.matrixPlot,
	}
};

const mapDispatchToProps = dispatch => {
    return {
        onFit: (data) => dispatch(fit(data)),
        onPredict: (data, from) => dispatch(predict(data, from)),
        onSplitDataSet: (data) => dispatch(splitDataSet(data)),
        updateData: data => dispatch(updateDataSuccess(data))
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(MachineLearning);