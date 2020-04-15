import React, { useState, useRef, useEffect } from 'react';
import CKeditor from '@ckeditor/ckeditor5-react';
import ClassicEditor from '@ckeditor/ckeditor5-build-classic';
import { MDBBtn, MDBIcon } from 'mdbreact';
import { convertToPDF } from '../../../utility/utility';
import jsPDF from 'jspdf';
import axios from './../../../instanceAxios';
import { connect } from 'react-redux';
import { NotificationContainer } from 'react-notifications';
import { createNotification } from './../../../utility/utility';



const Storytelling = (props) => {
    const [data, setData] = useState('');
    const [files, setFiles] = useState([]);
    const [error, setError] = useState(null);
    const node = useRef(null);

    const exportData = () => {
        let pdf = new jsPDF();
        pdf.setFontSize(18);
        pdf.fromHTML(node.current, 10, 10, {
            width: '100%' 
        });
        pdf.save("report.pdf");
    }

    useEffect(() => {
        axios.get('/api/plot-files/')
            .then(({data}) => {
                setFiles(data.files);
            }).catch(err => {
                setError(err.message);
                createNotification(err.message, 'Error');
            })
    }, []);


    return (
        <div>
            <div>
                <div className="row">
                    {
                        props.plots.map(plot => <div className="col-md-4 mt-3"><img style={{
                            width: '250px',
                            height: '250px'
                        }} src={plot} alt=""/></div>)
                    }
                    {
                        files.map(file => <div className="col-md-4 mt-3"><img style={{
                            width: '250px',
                            height: '250px'
                        }} src={file.url} alt=""/></div>)
                    }
                </div>
            </div>
            <div className="mt-3">
                    <CKeditor 
                    editor={ClassicEditor}
                    onInit={ editor => {
                        console.log('Editor is ready !');
                    }}

                    onChange={ (event, editor ) => {
                        setData(editor.getData());
                        console.log({ event, editor, data});
                    }}
                />
                <MDBBtn onClick={exportData}>
                    <MDBIcon className="mr-2" icon="download"/>Export Data
                </MDBBtn>
                <div ref={node} style={{
                    position: 'absolute',
                    zIndex: -1,
                }} dangerouslySetInnerHTML={{ __html: data }}>
                </div>
            </div>
            <NotificationContainer />
            </div>
    )
};

const mapStateToProps = state => {
    return {
        plots: state.statistics.plots
    }
}
 
export default connect(mapStateToProps)(Storytelling);