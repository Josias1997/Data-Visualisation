import React from 'react';
import { Switch, Route } from 'react-router-dom';
import DataManipulationPage from '../Pages/DataManipulation/DataManipulationPage';
import RStatisticsPage from '../Pages/RStatistics/RStatisticsPage';
import ModelisationPage from '../Pages/Modelisation/ModelisationPage';
import MachineLearning from '../Pages/MachineLearning/MachineLearning';
import DeepLearning from '../Pages/DeepLearning/DeepLearning';
import TextMining from '../Pages/TextMining/TextMining';
import Visualisation from '../Pages/Visualisation/Visualisation';

const Routes = props => {
    return (
		<Switch>
            <Route exact path={"/"} component={DataManipulationPage} />
            <Route path={"/r-statistics"} component={RStatisticsPage}/>
            <Route path={"/modelisation"} component={ModelisationPage} />
            <Route path={"/machine-learning"} component={MachineLearning} />
            <Route path={"/deep-learning"} component={DeepLearning} />
          	<Route path={"/text-mining"} component={TextMining} />
           	<Route path={"/visualisation"} component={Visualisation} />
   	 	</Switch>
    )
}
export default Routes;