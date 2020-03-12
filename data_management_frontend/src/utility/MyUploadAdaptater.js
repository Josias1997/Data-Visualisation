import axios from '../instanceAxios';

class MyUploadAdaptater {
    constructor( loader ) {
        // The file instance to use during upload. 
        // The loader will be passed into the adaptater later on.
        this.loader = loader;
    }

    // Start the upload process
    upload() {
        return this.loader.file
            .then(file => new Promise(( resolve, reject ) => {
                axios.post('/api/upload-editor-files/', {
                    'upload': file
                }).then(response => {
                    resolve({
                        default: response.url
                    });
                }).catch(error => {
                    reject(error);
                })
            }));
    }

    // Abort the upload process
    abort() {
        if (this.xhr) {
            this.xhr.abort();
        }
    }

}

export default MyUploadAdaptater;