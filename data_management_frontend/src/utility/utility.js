import axios from '../instanceAxios';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

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
	return table.map(element => element.field);
};

export const range = (startIndex = 0, endIndex) => {
	return [...Array(endIndex).keys()].map(index => index + startIndex);
};

export const getFunctionsList = () => {
	return ['log(x)', 'log10(x)', 'abs(x)', 'median(x)', 'quantile(x)', 'round(x)',
		'signif(x)', 'sin(x)', 'cos(x)', 'tan(x)', 'sqrt(x)', 'max(x)', 'min(x)', 'length(x)',
		'range(x)', 'sum(x)', 'prod(x)', 'mean(x)', 'var(x)', 'sort(x)', 'stdev(x)'
	];
};


export const convertToPDF = (algorithm, element) => {
	const filename = `results-${algorithm}.pdf`;

    html2canvas(element).then(canvas => {
        let pdf = new jsPDF('p', 'mm', 'a4');
        pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, 0, 211, 298);
        pdf.save(filename);
    });
};


const saveAs = (uri, filename) => {
	let link = document.createElement('a');

	if (typeof link.download === 'string') {
		link.href = uri;
		link.download = filename;

		document.body.appendChild(link);
		link.click();

		document.body.removeChild(link);
	} else {
		window.open(uri);
	}
}

export const convertToJPEG = (algorithm, element) => {
	html2canvas(element).then(canvas => {
		let imageData = canvas.toDataURL('image/jpeg', 'JPEG', 0, 0, 211, 298)
				saveAs(imageData, `results-${algorithm}.jpeg`);

	})
};

export const convertToPNG = (algorithm, element) => {
	html2canvas(element).then(canvas => {
		let imageData = canvas.toDataURL('image/png', 'PNG', 0, 0, 211, 298)
				saveAs(imageData, `results-${algorithm}.png`);

	})
};