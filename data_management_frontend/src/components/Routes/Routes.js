import React from 'react';
import { Switch, Route } from 'react-router-dom';
import DataManipulationPage from '../Pages/DataManipulation/DataManipulationPage';
import RStatisticsPage from '../Pages/RStatistics/RStatisticsPage';
import ModelisationPage from '../Pages/Modelisation/ModelisationPage';
import MachineLearning from '../Pages/MachineLearning/MachineLearning';

const Routes = props => {
    return (
		<Switch>
            <Route exact path={"/"} component={DataManipulationPage} />
            <Route path={"/r-statistics"} component={RStatisticsPage}/>
            <Route path={"/modelisation"} component={ModelisationPage} />
            <Route path={"/machine-learning"} component={MachineLearning} />
   	 	</Switch>
    )
}
export default Routes;