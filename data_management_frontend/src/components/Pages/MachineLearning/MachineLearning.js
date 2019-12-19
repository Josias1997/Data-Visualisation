import React, { useState, useEffect } from "react";
import { connect } from "react-redux";
import {options, addRow, updateRow, deleteRow} from '../../../utility/settings';
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

    useEffect(() => {
        if (props.fileId !== undefined) {
             const data = createJsonData(['id'], [props.fileId]);
             props.onSplitDataSet(data);
        }
    }, [props.fileId]);

    const handleAgorithmChange = (event) => {
        setAlgorithm(event.target.value);
    };

    const handleXChange = (event) => {
        setXValue(event.target.value);
    };

    const handleYChange = (event) => {
        setYValue(event.target.value);
    };

    const fit = () => {
        const data = createJsonData(['id', 'y'], [props.fileId, yValue]);
        props.onFit(data);
    };

    const predict = () => {
        const data = createJsonData(['id', 'x', 'y', 'algorithm'], [props.fileId, xValue, yValue, algorithm]);
        props.onPredict(data);
    };

	return (
		<MDBContainer>
			{
				props.fileId ? <>
				 <AlignCenter>
                    <MDBCol col={12}>
                        <Settings page="machine-learning" onFit={fit} onPredict={predict}/>
                    </MDBCol>
                </AlignCenter>
                <MDBCol col={12}>
                    <hr style={{
                        border: '2px solid #ccc',
                    }}/>
                </MDBCol>
                <div className="container justify-content-center mt-5 mb-3">
                {
                    (algorithm === 'multiple-linear-regression' || algorithm === 'logistic-regression' || algorithm === 'svr'
                    || algorithm === 'decision-tree-regressor' || algorithm === 'random-forest-regression') ? null : <>
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
                            <MenuItem value={'linear-regression'}>Linear Regression</MenuItem>
                            <MenuItem value={'multiple-linear-regression'}>Multiple Linear Regression</MenuItem>
                            <MenuItem value={'logistic-regression'}>Logistic Regression</MenuItem>
                            <MenuItem value={'svr'}>SVR</MenuItem>
                            <MenuItem value={'decision-tree-regressor'}>Decsion Tree Regressor</MenuItem>
                            <MenuItem value={'random-forest-regression'}>Random Forest Regression</MenuItem>
                        </Select>
                    </FormControl>
                </div>
                {
                (props.processing ? <Spinner /> : <>
                     <div className="container col-md-12 justify-content-center mt-5 mb-3">
                    {
                        props.predicted ? <> {
                            (algorithm === 'multiple-linear-regression' || algorithm === 'logistic-regression' || algorithm === 'svr'
                                 || algorithm === 'decision-tree-regressor' || algorithm === 'random-forest-regression') ? null : <>
                            <table className="table table-stripped">
                        <thead>
                            <tr>
                                <th>X_test</th>
                                <th>Predictions Y</th>
                            </tr>
                        </thead>
                        <tbody>
                        { props.predictResult !== undefined ? props.predictResult.map((arr, index) => <tr key={index}>
                            {
                                arr.map(value => <td key={$`{value} - {index}`}>{value}</td>)
                            }
                            </tr>) : null }
                        </tbody>
                        </table>
                            </>
                        } 

                        <div className="container" style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}>
                        {
                            algorithm === 'random-forest-regression' ? <>
                                 <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.randomForestRegressionGraph} alt="Random Forest Graph"/>
                                </div>
                            </> : null
                        }
                        {
                            algorithm === 'decision-tree-regressor' ? <>
                                 <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.decisionTreeGraphImg} alt="Decision Tree Graph"/>
                                </div>
                            </> : null
                        }
                        {
                            algorithm === 'svr' ? <>
                                 <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.svrResults} alt="SVR Results"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.svrResultsHR} alt="SVR Results with higher resolution" />
                                </div>
                            </> : null
                        }
                        {
                            algorithm === 'logistic-regression' ? <>
                                <h3>Confusion Matrix</h3>
                                <div className=" mt-2 container justify-content-center">
                                    {props.confusionMatrix !== undefined ? <Matrix matrix={props.confusionMatrix} /> : null}
                                </div>
                                <h3 className="mt-3">Confusion Matrix Plot</h3><br />
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.matrixPlot} alt="Confusion Matrix"/>
                                </div>
                                <h3 className="mt-3">Classification report</h3>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <span style={{
                                        whiteSpace: 'pre-wrap',
                                        fontWeight: 'bold'
                                    }}>{props.report}</span>
                                </div>
                                <h3 className="mt-5">Courbe ROC</h3>
                                <div className="col-md-12">
                                    <img src={props.courbeRoc} alt="Confusion Matrix"/>
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <h3>Score ROC : {props.scoreRoc}</h3>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.trainPlotPath} alt="Train set plot"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.testPlotPath} alt="Test set plot" />
                                </div>
                            </> : null
                        }

                        {
                            algorithm === 'linear-regression' ?  <> <div className="col-md-12">
                                <img src={props.trainPlotPath} alt="Train set plot"/>
                            </div>
                            <div className="col-md-12">
                                <img src={props.testPlotPath} alt="Test set plot" />
                            </div></> : null
                        }
                        {
                            algorithm === 'multiple-linear-regression' ? <>
                            <h2>Valeurs réelles vs Valeurs prévues</h2>
                            <div className="col-md-12">
                                <img src={props.seabornPlot} alt="Seaborn plot"/>
                            </div>
                            <h2>Nuage de points</h2>
                            <div className="col-md-12">
                                <img src={props.adminPlot} alt="Admin plot"/>
                            </div>
                            <div className="col-md-12">
                                <img src={props.marketingPlot} alt="Marketing plot" />
                            </div>
                            <div className="col-md-12">
                                <img src={props.rdSpendPlot} alt="RD Spend plot"/>
                            </div>
                            </> : null
                        }
                        </div>
                        </>
                        : null
                    }
                </div>
                {
                    props.predicted ? null : (props.splitProcessing ? <Spinner /> : <> 
                    <div className="container justify-content-center mt-5 mb-3">
                    <MaterialTable
                    title={"Training set"}
                    columns={props.trainingSet.columns}
                    data={props.trainingSet.rows}
                    options={options}
                    editable={{
                        onRowAdd: newData => addRow(newData),
                        onRowUpdate: (newData, oldData) => updateRow(newData, oldData),
                        onRowDelete: oldData => deleteRow(oldData)
                    }}
                    />
                </div>
        <div className="container justify-content-center mt-5 mb-3">
            <MaterialTable
            title={"Test set"}
            columns={props.testSet.columns}
            data={props.testSet.rows}
            options={options}
            editable={{
                onRowAdd: newData => addRow(newData),
                onRowUpdate: (newData, oldData) => updateRow(newData, oldData),
                onRowDelete: oldData => deleteRow(oldData)
            }}
            />
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
        fitResult: state.machine_learning.fitResult,
        predictResult: state.machine_learning.predictResult,
        predicted: state.machine_learning.predicted,
        trainPlotPath: state.machine_learning.trainPlotPath,
        testPlotPath: state.machine_learning.testPlotPath,
        xTest: state.machine_learning.xTest,
        splitProcessing: state.modelisation.processing,
        seabornPlot: state.machine_learning.seabornPlot,
        adminPlot: state.machine_learning.adminPlot,
        rdSpendPlot: state.machine_learning.rdSpendPlot,
        marketingPlot: state.machine_learning.marketingPlot,
        confusionMatrix: state.machine_learning.confusionMatrix,
        matrixPlot: state.machine_learning.matrixPlot,
        report: state.machine_learning.report,
        courbeRoc: state.machine_learning.courbeRoc,
        scoreRoc: state.machine_learning.scoreRoc,
        svrResults: state.machine_learning.svrResults,
        svrResultsHR: state.machine_learning.svrResultsHR,
        decisionTreeGraphImg: state.machine_learning.decisionTreeGraphImg,
        randomForestRegressionGraph: state.machine_learning.randomForestRegressionGraph,
	}
};

const mapDispatchToProps = dispatch => {
    return {
        onFit: (data) => dispatch(fit(data)),
        onPredict: (data) => dispatch(predict(data)),
        onSplitDataSet: (data) => dispatch(splitDataSet(data)),
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(MachineLearning);