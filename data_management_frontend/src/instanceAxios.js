import axios from 'axios';

const instance = axios.create({
	headers: {
		Accept: 'application/json, text/plain, */*',
        "Content-Type": "multipart/form-data"
	},
	xsrfCookieName: 'csrftoken',
	xsrfHeaderName: 'X-CSRFToken',
});

export default instance;