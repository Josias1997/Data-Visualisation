export {
    uploadFail,
    uploadSuccess,
    uploadStart,
    updateDataFail,
    updateDataStart,
    updateDataSuccess,
    removeFile,
    sendFile,
    onChangeHandler
} from './fileUpload';

export {
    columnInputValueChanged,
    handleColumnsChange,
    paramsForFilteringChanged,
    applySettings,
    columnsAndRowsValueChangeHandler,
} from './parameters';

export {
    authStart,
    authFail,
    authLogin,
    authCheckState,
    authSuccess,
    checkAuthTimeOut,
    logout,
} from './auth';

export {
    openTable,
    openPlot,
    openTests
} from './statistics';

export {
    testFail,
    testSuccess,
    startTest,
    test
} from "./tests";

export {
    startPreprocessing,
    preprocessingDataSplitSuccess,
    preprocessingNormalizingSuccess,
    preprocessingFail,
    splitDataSet,
    normalize,

} from "./modelisation";

export {
    processingStart,
    fitSuccess,
    predictSuccess,
    processingFail,
    fit,
    predict

} from "./machine_learning";