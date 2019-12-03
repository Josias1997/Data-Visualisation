import React, {useState, useRef, useEffect} from "react";
import { createJsonData } from "../../../utility/utility";
import axios from "../../../instanceAxios";

const CustomDataTable = ({columns, rows, updateResult}) => {
    const [testTable, setTestTable] = useState([]);
    const [startTest, setStartTest] = useState(false);
    const dataCount = useRef(0);
    const refs = useRef([]);

    useEffect(() => {
        if(startTest) {
            fisherTest();
            setStartTest(false);
        }
    }, [startTest])

    const selectValue = (index) => {
        const refValue = refs.current[index];
        if(testTable.indexOf(refValue.innerText) !== -1) {
            refValue.style.color = "initial";
            refValue.style.backgroundColor = "transparent";
            setTestTable(currentValues => {
                currentValues.splice(testTable.indexOf(refValue.innerText), 1);
                return currentValues;
            })
        }
        else if (testTable.length < 4) { 
            refValue.style.color = "white";
            refValue.style.backgroundColor = "green";
            setTestTable(currentValues => {
                currentValues.push(refValue.innerText);
                return currentValues;
            })
            if (testTable.length === 4) {
                setStartTest(true);
            }
        }
    }
    const fisherTest = () => {
        const data = createJsonData(['table'], [testTable]);
        axios.post('/api/fisher-test/', data)
        .then(response => {
            updateResult(response.data);
        }).catch(error => {
            updateResult({
                result: '',
                error: error.message
            })
        })
    };

    return (
        <table className="table table-bordered table-hover table-responsive nowrap">
            <thead>
                <th>#</th>
                {
                    columns.map(column => <th key={column.label}>{column.label}</th>)
                }
            </thead>
            <tbody>
                {
                    rows.map((row, index) => {
                        return <tr key={index}>
                            <td><strong>{index + 1}</strong></td>
                            {
                                columns.map(column => {
                                    let count = dataCount.current++;
                                    return <td key={`${column.label}${index}`} 
                                        ref={el => refs.current[count] = el} 
                                        onClick={() => selectValue(count)}
                                        style={{
                                            cursor: 'pointer'
                                        }}
                                    >
                                        {row[column.label]}
                                    </td>
                                })
                            }
                        </tr>
                    })
                }
            </tbody>
        </table>
    );
};

export default CustomDataTable;