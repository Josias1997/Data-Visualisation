import React, {useState} from 'react';
import axios from '../../instanceAxios';
import CSRFToken from '../../utils/CSRFToken.jsx';
import Spinner from './Spinner.jsx';

const FileForm = props => {
    const [file, setFile] = useState('');
    const [data, setData] = useState('');
    const [columnsNames, setColumnsNames] = useState([]);
    const [loading, setLoading] = useState(false);
    const [fileId, setFileId] = useState('');
    const [beginLine, setBeginLine] = useState('');
    const [endLine, setEndLine] = useState('');
    const [beginColumn, setBeginColumn] = useState('');
    const [endColumn, setEndColumn] = useState('');
    const [rows, setRows] = useState(0);
    const [columns, setColumns] = useState(0);
    const [size, setSize] = useState(0);
    const [searchInput, setSearchInput] = useState('');
    const [columnToTransform, setColumnToTransform] = useState('');
    const [type, setType] = useState('');
    const [query, setQuery] = useState('');
    const [selectedColumns, setSelectedColumns] = useState([]);

    const onChangeHandler = event => {
        setFile(event.target.files[0]);
    };
    const sendFile = () => {
        const data = new FormData();
        data.append('file', file);
        setLoading(true);
        axios.post('/api/upload/', data)
            .then(response => {
                setData(response.data.data_table);
                setColumnsNames(response.data.columns_name);
                setFileId(response.data.id);
                setRows(response.data.rows);
                setColumns(response.data.columns);
                setSize(response.data.size);
                setLoading(false);
                setSelectedColumns([]);
            }).catch(error => {
            setData("Erreur chargement du fichier. Veuillez reprendre l'upload.");
            setLoading(false);
        })
    };
    const styles = {
        width: '100%'
    };

    const onChangeFilterParamsHandler = event => {
        const value = event.target.value;
        switch (event.target.id) {
            case 'beginLine':
                setBeginLine(value);
                break;
            case 'endLine':
                setEndLine(value);
                break;
            case 'beginColumn':
                setBeginColumn(value);
                break;
            case 'endColumn':
                setEndColumn(value);
                break;
            case 'searchInput':
                setSearchInput(value);
                break;
            case 'columnInput':
                setColumnToTransform(value);
                break;
            case 'options':
                setType(value);
                break;
            case 'query':
                setQuery(value);
                break;

        }
    };

    const filterData = () => {
        const data = new FormData();
        data.append('id', fileId);
        data.append('beginLine', beginLine);
        data.append('endLine', endLine);
        data.append('beginColumn', beginColumn);
        data.append('endColumn', endColumn);
        setLoading(true);
        axios.post('/api/filter/', data)
            .then(response => {
                setData(response.data);
                setLoading(false);
            }).catch(error => {
            setData("Echec filtrage");
            setLoading(false);
        })
    };

    const onSearchHandler = () => {
        const data = new FormData();
        data.append('id', fileId);
        data.append('value', searchInput);
        setLoading(true);
        axios.post('/api/search/', data)
            .then(response => {
                setData(response.data);
                setLoading(false);
            }).catch(error => {
            setData("Echec de la recherche veuillez reprendre svp!");
            setLoading(false);
        })
    };

    const describeHandler = () => {
        const data = new FormData();
        data.append('id', fileId);
        setLoading(true);
        axios.post('/api/describe/', data)
            .then(response => {
                setData(response.data);
                setLoading(false);
            }).catch(error => {
            setData("Erreur serveur veuillez réessayer svp!");
            setLoading(false);
        })
    };

    const transformHandler = () => {
        const data = new FormData();
        data.append('id', fileId);
        data.append('column', columnToTransform);
        data.append('type', type);
        setLoading(true);
        axios.post('/api/transform/', data)
            .then(response => {
                setData(response.data);
                setLoading(false);
            }).catch(error => {
            setData("Erreur serveur veuillez réessayer svp!");
            setLoading(false);
        })
    };

    const queryHandler = () => {
        const data = new FormData();
        data.append('id', fileId);
        data.append('query', query);
        setLoading(true);
        axios.post('/api/execute-query/', data)
            .then(response => {
                setData(response.data);
                setLoading(false);
            }).catch(error => {
            setData("Echec requête veuillez réessayer svp!");
            setLoading(false);
        })
    };

    const filterColumns = () => {
        const data = new FormData();
        data.append('id', fileId);
        data.append('columns_names', selectedColumns);
        setLoading(true);
        axios.post('/api/filter-columns/', data)
            .then(response => {
                setData(response.data);
                setLoading(false);
            }).catch(error => {
            setData("Echec requête veuillez réessayer svp!");
            setLoading(false);
        })
    };

    const handleColumnsChange = event => {
        const target = event.target;
        if (target.checked) {
            const names = [...selectedColumns];
            setSelectedColumns(currentCols => {
                currentCols.push(target.id);
                return currentCols;
            })
        } else {
            setSelectedColumns(currentCols => {
                if (currentCols.includes(target.id)) {
                    currentCols.splice(currentCols.indexOf(target.id), 1);
                }
                return currentCols;
            })
        }
        console.log(selectedColumns);
    };
    return (
        <div className="row justify-content-center">
            <form className="col-md-5 mt-5" method="POST" encType="multipart/form-data">
                <CSRFToken/>
                <div className="input-group ml-2">
                    <div className="custom-file mt-2">
                        <input
                            type="file"
                            id="inputGroupFile01"
                            aria-describedby="inputGroupFileAddon01"
                            onChange={onChangeHandler}
                        />
                        <label className="custom-file-label" htmlFor="inputGroupFile01">
                            Parcourir
                        </label>
                    </div>
                    <a type="submit" className="btn purple-gradient mb-2" onClick={sendFile}>Charger</a>
                </div>
            </form>
            {
                fileId ? <div className="col-md-10 d-flex mb-4 justify-content-center card  purple lighten-4">
                    <div className="container row">
                        <div className="col-md-12 d-flex justify-content-center mt-4">
								<span className="badge badge-info">
									{`Enregistrements: ${size}, Nombre de lignes: ${rows}, Nombre de colonnes: ${columns}`}
								</span>
                            <button className={"purple-gradient"} onClick={describeHandler}>Statistiques descriptives</button>
                        </div>
                        <div className="col-md-4 mt-5">
                            <form>
                                <input id="beginLine" type="number" min={0} className="form-control"
                                       placeholder="Début lignes" onChange={onChangeFilterParamsHandler}/>
                                <input id="endLine" type="number" min={0} className="form-control"
                                       placeholder="Fin Lignes" onChange={onChangeFilterParamsHandler}/>
                                <input id="beginColumn" type="number" min={0} className="form-control"
                                       placeholder="Début colonnes" onChange={onChangeFilterParamsHandler}/>
                                <input id="endColumn" type="number" min={0} className="form-control"
                                       placeholder="Fin colonnes" onChange={onChangeFilterParamsHandler}/>
                                <a type="submit" className="btn purple-gradient" onClick={filterData}>Filtrer</a>
                            </form>
                        </div>
                        <div className="col-md-4 mt-5">
                            <form className="form-inline">
                                <i className="fas fa-search"
                                   aria-hidden="true"
                                   style={{
                                       cursor: 'pointer',
                                   }}
                                   onClick={onSearchHandler}
                                >
                                </i>
                                <input className="form-control form-control-sm ml-3 w-75" type="text"
                                       placeholder="Recherche"
                                       aria-label="Search"
                                       id={"searchInput"}
                                       onChange={onChangeFilterParamsHandler}
                                />
                            </form>
                            <input className="form-control mt-3" type="text"
                                   placeholder="Requête SQL"
                                   aria-label="Search"
                                   id={"query"}
                                   onChange={onChangeFilterParamsHandler}
                            />
                            <button className="btn purple-gradient" onClick={queryHandler}>Exécuter</button>
                        </div>
                        <div className={"col-md-4 mt-5"}>
                            <input className="form-control" type="text"
                                   placeholder="Colonne"
                                   aria-label="Search"
                                   id={"columnInput"}
                                   onChange={onChangeFilterParamsHandler}
                            />
                            <select id={'options'} className="browser-default custom-select"
                                    onChange={onChangeFilterParamsHandler}>
                                <option defaultValue={""}>Type conversion</option>
                                <option value="int">Int</option>
                                <option value="float">Float</option>
                                <option value="str">String</option>
                            </select>
                            <button className="btn purple-gradient" onClick={transformHandler}>Transformer</button>
                        </div>
                        <div className={"col-md-12 d-flex justify-content-center mb-4"}>
                            {
                                columnsNames.map(name => {
                                    return <div className="custom-control custom-checkbox ml-1" key={name}>
                                        <input type="checkbox" className="custom-control-input" id={name}
                                               onChange={handleColumnsChange}/>
                                        <label className="custom-control-label" htmlFor={name}>{name}</label>
                                    </div>;
                                })
                            }
                            <i className={"fas fa-check-circle ml-2 mt-1"} onClick={filterColumns} style={{
                                cursor: 'pointer'
                            }}>

                            </i>
                        </div>
                    </div>
                </div> : null
            }

            <div className="container d-flex justify-content-center">
                {
                    !loading ? <div style={{
                        ...styles
                    }} dangerouslySetInnerHTML={{
                        __html: data
                    }}/> : <Spinner/>
                }
            </div>
        </div>
    );
};


export default FileForm;