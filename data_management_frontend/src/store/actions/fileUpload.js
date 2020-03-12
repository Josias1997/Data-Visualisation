import * as actionTypes from './actionTypes';
import axios from './../../instanceAxios';

export const uploadStart = () => {
    return {
        type: actionTypes.UPLOAD_START
    }
};

export const uploadSuccess = details => {
    return {
        type: actionTypes.UPLOAD_SUCCESS,
        details: details
    }
};

export const uploadFail = error => {
    return {
        type: actionTypes.UPLOAD_FAIL,
        error: error
    }
};

export const updateDataStart = () => {
    return {
        type: actionTypes.UPDATE_DATA_START
    }
};

export const updateDataSuccess = data => {
    return {
       type: actionTypes.UPDATE_DATA_SUCCESS,
       data: data,
    }
};

export const updateDataFail = error => {
    return {
        type: actionTypes.UPDATE_DATA_FAIL,
        error: error
    }
};

export const removeFile = () => {
    axios.delete('/api/delete-files/')
        .then(({data}) => console.log(data)).catch(err => console.err);
    return {
        type: actionTypes.REMOVE_FILE
    }
};

export const inputValueChanged = file => {
    return {
        type: actionTypes.INPUT_VALUE_CHANGED,
        file: file
    }
};

export const dataInfos = data => {
    return {
        type: actionTypes.DF_INFO,
        data: data,
    }
};

export const resetData = data => {
    return {
        type: actionTypes.RESET_DATA,
        data: data
    }
};

export const sendFile = file => {
    return dispatch => {
        dispatch(uploadStart());
        const data = new FormData();
        data.append('file', file);
        axios.post('/api/upload/', data)
            .then(response => {
                dispatch(uploadSuccess(response.data))
            }).catch(error => {
            dispatch(uploadFail(error));
        })
    }
};

export const onChangeHandler = event => {
    return dispatch => {
        let file = event.target.files[0];
        dispatch(inputValueChanged(file))
    }
};

export const getInfos = (id) => {
    return dispatch => {
        dispatch(uploadStart());
        const data = new FormData();
        data.append('id', id);
        axios.post('/api/info/', data)
        .then(({data}) => {
            dispatch(dataInfos(data));
        }).catch(error => {
            dispatch(uploadFail(error));
        });
    }
}

export const reset = (id) => {
    return dispatch => {
        dispatch(uploadStart());
        const data = new FormData();
        data.append('id', id);
        axios.post('/api/reset/', data)
        .then(response => {
            dispatch(resetData(response.data));
        }).catch(error => {
            dispatch(uploadFail(error));
        });
    }
}