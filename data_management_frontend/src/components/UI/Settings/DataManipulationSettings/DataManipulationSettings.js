import React from 'react';
import Statistics from "./Statictics/Statistics";
import SQLQueryForm from "./SQLQueryForm/SQLQueryForm";
import ConversionForm from "./ConversionForm/ConversionForm";

const DataManipulationSettings = props => {
    return (
        <>
            <Statistics />
            <SQLQueryForm/>
        </>
    );
}

export default DataManipulationSettings;