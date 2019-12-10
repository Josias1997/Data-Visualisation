import React from 'react';
import { Switch, Route } from 'react-router-dom';
import DataManipulationPage from '../Pages/DataManipulation/DataManipulationPage';
import RStatisticsPage from '../Pages/RStatistics/RStatisticsPage';
import ModelisationPage from '../Pages/Modelisation/ModelisationPage';
import MachineLearning from '../Pages/MachineLearning/MachineLearning';
import ErrorBoundary from "../HOC/ErrorBoundary/ErrorBoundary";

const Routes = props => {
    return (
    	<ErrorBoundary>
    		 <Switch>
	            <Route exact path={"/"} component={DataManipulationPage} />
	            <Route path={"/r-statistics"} component={RStatisticsPage}/>
	            <Route path={"/modelisation"} component={ModelisationPage} />
	            <Route path={"/machine-learning"} component={MachineLearning} />
       	 	</Switch>
    	</ErrorBoundary>
    )
}
export default Routes;