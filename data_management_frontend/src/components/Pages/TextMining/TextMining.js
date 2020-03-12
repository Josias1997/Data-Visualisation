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

    const fit = () => {
        const data = createJsonData(['id', 'y'], [props.fileId, yValue]);
        props.onFit(data);
    };

    const predict = () => {
        const data = createJsonData(['id', 'x', 'y', 'algorithm'], [props.fileId, xValue, yValue, algorithm]);
        props.onPredict(data, 'text_mining');
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
                    (algorithm === 'sentimental-analysis') 
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
                            <MenuItem value={'sentimental-analysis'}>Sentimental Analysis</MenuItem>
                        </Select>
                    </FormControl>
                </div>
                {
                (props.processing ? <Spinner /> : <>
                     <div className="container col-md-12 justify-content-center mt-5 mb-3">
                    {
                        props.predicted ? <> {
                            (algorithm === 'sentimental-analysis') 
                            ? null : <>
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
                                algorithm === 'sentimental-analysis' ? <>

                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.retweets} alt="Confusion Matrix"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.languages_used} alt="Train set plot"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.original_authors_retweets} alt="Test set plot" />
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.correlation_matrix} alt="Confusion Matrix"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.tweets_by_month} alt="Train set plot"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.popular_hashtags_used} alt="Test set plot" />
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.hsh_wrds_trump} alt="Confusion Matrix"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.hsh_wrds_hillary} alt="Train set plot"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.popular_twitter_account_references} alt="Test set plot" />
                                </div>
                                <div className="col-md-12 d-flex justify-content-center">
                                    <img src={props.acc_wrds_trump} alt="Confusion Matrix"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.acc_wrds_hillary} alt="Train set plot"/>
                                </div>
                                <div className="col-md-12">
                                    <img src={props.pop_wrds_trump} alt="Test set plot" />
                                </div>
                                <div className="col-md-12">
                                    <img src={props.pop_wrds_hillary} alt="Test set plot" />
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
        processing: state.text_mining.loading,
        fitResult: state.machine_learning.fitResult,
        predictResult: state.machine_learning.predictResult,
        predicted: state.text_mining.predicted,
        splitProcessing: state.modelisation.processing,
        retweets: state.text_mining.retweets,
        languages_used: state.text_mining.languages_used,
        original_authors_retweets: state.text_mining.original_authors_retweets,
        tweets_by_month: state.text_mining.tweets_by_month,
        correlation_matrix: state.text_mining.correlation_matrix,
        popular_hashtags_used: state.text_mining.popular_hashtags_used,
        hsh_wrds_trump: state.text_mining.hsh_wrds_trump,
        hsh_wrds_hillary: state.text_mining.hsh_wrds_hillary,
        popular_twitter_account_references: state.text_mining.popular_twitter_account_references,
        acc_wrds_trump: state.text_mining.acc_wrds_trump,
        acc_wrds_hillary: state.text_mining.acc_wrds_hillary,
        pop_wrds_trump: state.text_mining.pop_wrds_trump,
        pop_wrds_hillary: state.text_mining.pop_wrds_hillary,
        popular_negative_words: state.text_mining.popular_negative_words,
        pw_trump: state.text_mining.pw_trump,
        pw_hillary: state.text_mining.pw_hillary,
        nw_trump: state.text_mining.nw_trump,
        nw_hillary: state.text_mining.nw_hillary,
        sentiment_of_tweets: state.text_mining.sentiment_of_tweets,
        average_retweets: state.text_mining.average_retweets,
        classifier_trump: state.text_mining.classifier_trump,
        classifier_hillary: state.text_mining.classifier_hillary,
        tweets_trump: state.text_mining.tweets_trump,
        tweets_hillary: state.text_mining.tweets_hillary,
        log_reg_trump: state.text_mining.log_reg_trump,
        log_reg_hillary: state.text_mining.log_reg_hillary ,
        svm_trump: state.text_mining.svm_trump,
        svm_hillary: state.text_mining.svm_hillary,
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