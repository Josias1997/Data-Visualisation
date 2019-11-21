import axios from '../instanceAxios';

export const updateObject = (oldObject, updatedProperties) => {
	return {
		...oldObject,
		...updatedProperties,
	}
};

export const createJsonData = (keys, values) => {
	const data = new FormData();
	for(let i = 0; i < values.length; i++) {
		data.append(keys[i], values[i]);
	}
	return data;
};

export const postData = (url, data, ) => {
	axios.post(url, data)
		.then(response => {
			return response.data;
		}).catch(error => {
			return error;
	})
};

export const formatErrorData = (errorMessage) => {
	return {
		columns: [
			{
				name: 'error',
				label: 'Error',
			}
		],
		rows: [
			{
				error: errorMessage
			}
		]
	};
};