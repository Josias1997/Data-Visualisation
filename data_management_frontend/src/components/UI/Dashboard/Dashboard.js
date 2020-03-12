import React, { useState, useEffect, useRef } from 'react';
import Radium from 'radium';
import './Dashboard.css';
import axios from './../../../instanceAxios';
import Spinner from './../Spinner/Spinner';
import AlignCenter from './../AlignCenter/AlignCenter';

const Dashboard = (props) => {
    const el = useRef(null);
    const [fileToUpload, setFileToUpload] = useState('');
    const [currentFilesUrls, setCurrentFilesUrls] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        axios.get('/api/plot-files/')
            .then(({data}) => {
                setCurrentFilesUrls(data.urls);
            }).catch(err => {
                console.log(err);
        });
    }, []);

    const handleChange = (event) => {
        event.preventDefault();
        upload(event.target.files[0]);
    };
    let style = {
        border: '1px solid black',
        borderRadius: '30px',
        width: '300px',
        height: '250px',
        cursor: 'pointer',
        background: 'rgba(0, 0, 0, 0.3)',
        ':hover': {
            background: 'gray'
        }  
    };

    const handleDragEnter = (event) => {
        event.preventDefault();
        console.log("Drag enter", event);
        el.current.style.background = 'gray';
    };

    const handleDragLeave = (event) => {
        event.preventDefault();
        console.log("Drag Leave", event);
        el.current.style.background = 'rgba(0, 0, 0, 0.3)';
    }

    const handleDragOver = (event) => {
        event.preventDefault();
    }

    const handleDrop = (event) => {
        event.preventDefault();
        const supportedFilesTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'];
        const { type } = event.dataTransfer.files[0];
        console.log("On Drop", event);
        if (supportedFilesTypes.indexOf(type) !== -1) {
            upload(event.dataTransfer.files[0]);
        }
    };

    const upload = file => {
        const data = new FormData();
        data.append('file', file);
        setLoading(true);
        axios.post('/api/upload-plot-files/', data)
            .then(({data}) => {
                setCurrentFilesUrls(data.urls);
                setLoading(false);
            }).catch(err => {
                console.log(err);
        })
    };

    return (
        <div className="col-md-12" style={{
            border: '1px solid black',
            padding: '20px',
            borderRadius: '20px'
        }}>
            <div className="row">
            {
                currentFilesUrls.map(url => <div className="col-md-4 mt-3"><img src={url} alt="" style={{
                    width: '300px',
                    height: '300px'
                }}/></div>)
            }
             <div className="col-md-4 mt-3" >
                {
                    loading ? <Spinner /> :
                    <><label htmlFor="add-file" style={style} ref={el}>
                        <p  onDragEnter={handleDragEnter}
                            onDragLeave={handleDragLeave} 
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            style={{
                                position: 'absolute',
                                top: '20%',
                                left: '33%',
                                fontWeight: 'bold',
                                fontSize: '100px',
                                color: 'white'
                        }}><i className="fas fa-plus"></i></p>
                    </label></>
                }
                    <input id="add-file" className="mt-5" type="file" onChange={handleChange} accept="image/*" style={{
                        display: "none"
                    }} />
                </div>
            </div>
        </div>
    )
};

export default Radium(Dashboard);