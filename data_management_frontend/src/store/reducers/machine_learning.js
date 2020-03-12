import * as actionTypes from "./../actions/actionTypes";
import { updateObject } from "../../utility/utility";

const initialState = {
	fitResult: [],
	predictResult: [],
	loading: false,
	error: false,
	predicted: false,
	trainPlotPath: '',
	testPlotPath: '',
	seabornPlot: '',
	marketingPlot: '',
	rdSpendPlot: '',
	adminPlot: '',
	confusionMatrix: [],
	matrixPlot: '',
	report: '',
	courbeRoc: '',
	scoreRoc: '',
	svrResults: '',
	svrResultsHR: '',
	decisionTreeGraphImg: '',
	randomForestRegressionGraph: '',
	clusters: '',
	elbowGraph: '',
	dendrogramGraph: '',
	supportConfidence: '',
	supportLift: '',
	liftConfidence: '',
	histogram: '',
}

const processingStart = (state) => {
	return updateObject(state, {
		loading: true,
	});
};

const fitSuccess = (state, action) => {
	return updateObject(state, {
		fitResult: action.data.fit_result,
		loading: false
	});
};

const predictSuccess = (state, action) => {
	return updateObject(state, {
		predictResult: action.data.predict_result,
		trainPlotPath: action.data.train_plot,
		testPlotPath: action.data.test_plot,
		seabornPlot: action.data.seaborn_plot,
		marketingPlot: action.data.marketing_plot,
		rdSpendPlot: action.data.rd_spend_plot,
		adminPlot: action.data.admin_plot,
		confusionMatrix: action.data.confusion_matrix,
		matrixPlot: action.data.matrix_plot,
		report: action.data.report,
		courbeRoc: action.data.courbe_roc,
		scoreRoc: action.data.score_roc,
		svrResults: action.data.svr_results,
		svrResultsHR: action.data.svr_results_hr,
		decisionTreeGraphImg: action.data.decision_tree_graph_img,
		randomForestRegressionGraph: action.data.rdm_forest_regression_graph,
		clusters: action.data.clusters,
		elbowGraph: action.data.elbow_graph,
		dendrogramGraph: action.data.dendrogram,
		supportConfidence: action.data.support_confidence,
		supportLift: action.data.support_lift,
		liftConfidence: action.data.lift_confidence,
		histogram: action.data.histogram,
		loading: false,
		predicted: true,
	});
};

const processingFail = (state, action) => {
	return updateObject(state, {
		error: action.error,
		loading: false
	})
}

const reducer = (state = initialState, action) => {
	switch(action.type) {
		case actionTypes.PROCESSING_START_MACHINE_LEARNING:
			return processingStart(state);
		case actionTypes.FIT_SUCCESS:
			return fitSuccess(state, action);
		case actionTypes.PREDICT_SUCCESS_MACHINE_LEARNING:
			return predictSuccess(state, action);
		case actionTypes.PROCESSING_FAIL:
			return processingFail(state, action);
		default:
			return state;
	}
};

export default reducer;