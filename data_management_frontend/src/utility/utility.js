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

export const mapDataColumns = table => {
	return table.map(element => element.label);
};

export const range = (startIndex = 0, endIndex) => {
	return [...Array(endIndex).keys()].map(index => index + startIndex);
};

export const getFunctionsList = () => {
	return ['log(x)', 'log10(x)', 'abs(x)', 'median(x)', 'quantile(x)', 'round(x)',
		'signif(x)', 'sin(x)', 'cos(x)', 'tan(x)', 'sqrt(x)', 'max(x)', 'min(x)', 'length(x)',
		'range(x)', 'sum(x)', 'prod(x)', 'mean(x)', 'var(x)', 'sort(x)'
	];
};

