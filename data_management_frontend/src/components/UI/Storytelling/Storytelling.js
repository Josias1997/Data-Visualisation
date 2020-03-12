import React, { useState, useRef, useEffect } from 'react';
import CKeditor from '@ckeditor/ckeditor5-react';
import ClassicEditor from '@ckeditor/ckeditor5-build-classic';
import { MDBBtn, MDBIcon } from 'mdbreact';
import { convertToPDF } from '../../../utility/utility';
import jsPDF from 'jspdf';
import axios from './../../../instanceAxios';



const Storytelling = () => {
    const [data, setData] = useState('');
    const [urls, setUrls] = useState([]);
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
                setUrls(data.urls);
            }).catch(err => {
                console.log(err);
            })
    }, []);

    return (
        <div>
            <div>
                <div className="row">
                    {
                        urls.map(url => <div className="col-md-4 mt-3"><img style={{
                            width: '250px',
                            height: '250px'
                        }} src={url} alt=""/></div>)
                    }
                </div>
            </div>
            <div className="mt-3">
                    <CKeditor 
                    editor={ClassicEditor}
                    onInit={ editor => {
                        console.log('Editor is ready !', editor)
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
            </div>
    )
};


export default Storytelling;