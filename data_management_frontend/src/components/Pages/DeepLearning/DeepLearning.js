import React, { useState, useEffect } from "react";
import { connect } from "react-redux";
import { options } from '../../../utility/settings';
import { createJsonData } from '../../../utility/utility';
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
        props.onPredict(data);
    };

    const split = () => {
        if (props.fileId !== undefined) {
            const data = createJsonData(['id'], [props.fileId]);
            props.onSplitDataSet(data);
        }
    }
	return (
		<MDBContainer>
			{
				props.fileId ? <>
				 <AlignCenter>
                    <MDBCol col={12}>
                        <Settings page="machine-learning" onFit={fit} onPredict={predict} onSplit={split}/>
                    </MDBCol>
                </AlignCenter>
                <MDBCol col={12}>
                    <hr style={{
                        border: '2px solid #ccc',
                    }}/>
                </MDBCol>
                <div className="container justify-content-center mt-5 mb-3">
                {
                    (algorithm === 'fp-growth' || algorithm === 'apriori' || algorithm === 'thompson-sampling'
                    || algorithm === 'upper-confidence-bound' || algorithm === 'artificial-neural-network') 
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
                            <MenuItem value={'fp-growth'}>FP-Growth</MenuItem>
                            <MenuItem value={'apriori'}>A priori</MenuItem>
                            <MenuItem value={'thompson-sampling'}>Thompson Sampling</MenuItem>
                            <MenuItem value={'upper-confidence-bound'}>Upper Confidence Bound</MenuItem>
                            <MenuItem value={'artificial-neural-network'}>Artificial Neural Network</MenuItem>
                        </Select>
                    </FormControl>
                </div>
                {
                (props.processing ? <Spinner /> : <>
                     <div className="container col-md-12 justify-content-center mt-5 mb-3">

                        <div className="container" style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}>
                        {
                            algorithm === 'fp-growth' || algorithm === 'apriori' ? <>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.supportConfidence} alt="Confusion Matrix"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.supportLift} alt="Train set plot"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.liftConfidence} alt="Test set plot" />
                                </div>
                            </> : null
                        }
                        {
                            algorithm === 'thompson-sampling' || algorithm === 'upper-confidence-bound' ? <>
                                 <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.histogram} alt="Decision Tree Graph"/>
                                </div>
                            </> : null
                        }
                        {
                                algorithm === 'artificial-neural-network' ? <>
                                <h3>Confusion Matrix</h3>
                                <div className=" mt-2 container justify-content-center">
                                    {props.confusionMatrix !== undefined ? <Matrix matrix={props.confusionMatrix} /> : null}
                                </div>
                                <h3 className="mt-3">Confusion Matrix Plot</h3><br />
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.matrixPlot} alt="Confusion Matrix"/>
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
        processing: state.machine_learning.loading,
        predicted: state.machine_learning.predicted,
        splitProcessing: state.modelisation.processing,
        histogram: state.machine_learning.histogram,
        supportConfidence: state.machine_learning.supportConfidence,
        supportLift: state.machine_learning.supportLift,
        liftConfidence: state.machine_learning.liftConfidence,
        matrixPlot: state.machine_learning.matrixPlot,
        confusionMatrix: state.machine_learning.confusionMatrix,
	}
};

const mapDispatchToProps = dispatch => {
    return {
        onFit: (data) => dispatch(fit(data)),
        onPredict: (data) => dispatch(predict(data)),
        onSplitDataSet: (data) => dispatch(splitDataSet(data)),
        updateData: data => dispatch(updateDataSuccess(data))
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(MachineLearning);