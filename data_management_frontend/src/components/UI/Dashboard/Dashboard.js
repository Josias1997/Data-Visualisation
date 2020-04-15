import React, { useState, useEffect, useRef } from 'react';
import Radium from 'radium';
import './Dashboard.css';
import axios from './../../../instanceAxios';
import Spinner from './../Spinner/Spinner';
import AlignCenter from './../AlignCenter/AlignCenter';
import { connect } from 'react-redux';
import { createNotification } from './../../../utility/utility';
import { NotificationContainer } from 'react-notifications';
import { MDBCardGroup, MDBCard, MDBCardImage, MDBCardBody } from 'mdbreact';

const Dashboard = (props) => {
    const el = useRef(null);
    const [fileToUpload, setFileToUpload] = useState('');
    const [currentFilesUrls, setCurrentFilesUrls] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        axios.get('/api/plot-files/')
            .then(({data}) => {
                setCurrentFilesUrls(data.files);
            }).catch(err => {
                setError(err.message);
                createNotification(err.message, 'Error');
            });
    }, []);

    const handleChange = (event) => {
        event.preventDefault();
        upload(event.target.files[0]);
    };
    let style = {
        borderRadius: '10px',
        width: '100px',
        height: '150px',
        marginLeft: '70px',
        cursor: 'pointer'
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
                setCurrentFilesUrls(data.files);
                setLoading(false);
            }).catch(err => {
                setError(err.message);
                createNotification(err.message, 'Error');
            })
    };
    const deleteFile = id => {
        setLoading(true);
        axios.delete(`/api/delete-plot-file/${id}`)
        .then(({data}) => {
            setCurrentFilesUrls(current => {
                const index = current.findIndex(file => file.id === id);
                if (index !== -1) {
                    const currentFiles = [...current];
                    currentFiles.splice(index, 1);
                    return currentFiles;
                }
                return current;
            })
        }).catch(err => {
            setError(err.message);
            createNotification(err.message, 'Error');
        })
    }

    return (
        <div className="col-md-12">
        <MDBCardGroup>
            {
                props.plots.map(plot => <div className="col-md-4"><MDBCard><MDBCardImage src={plot} alt="" top hover overlay={"white-slight"}/></MDBCard></div>)
            }
            {
                currentFilesUrls.map(file => <div className="col-md-4"><MDBCard><MDBCardImage src={file.url} alt="" top hover overlay={"white-slight"}/></MDBCard></div>)
            }
            <div className="col-md-4">
             <MDBCard>
                <MDBCardBody>
                {
                    loading ? <Spinner /> :
                    <><label htmlFor="add-file" style={style} ref={el}>
                        <p  onDragEnter={handleDragEnter}
                            onDragLeave={handleDragLeave} 
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            style={{
                                fontWeight: 'bold',
                                fontSize: '100px',
                                color: '#2bbbad',
                                ':hover': {
                                    fontSize: '120px'
                                }
                        }}><i className="fas fa-plus"></i></p>
                    </label></>
                }
                    <input id="add-file" className="mt-5" type="file" onChange={handleChange} accept="image/*" style={{
                        display: "none"
                    }} />
                </MDBCardBody>
            </MDBCard>
            </div>
            <NotificationContainer />
        </MDBCardGroup>
        </div>
    )
};

const mapStateToProps = state => {
    return {
        plots: state.statistics.plots
    }
}

export default connect(mapStateToProps)(Radium(Dashboard));