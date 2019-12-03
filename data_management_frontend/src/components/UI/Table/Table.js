import React, {useRef, useState} from "react";
import { range } from "../../../utility/utility";


const Table = ({ columns, rows }) => {
	const dataCount = useRef(0);
	const refs = useRef([]);
	const [numberRowsPerPage, setNumberRowsPerPage] = useState(10);
	const [currentPage, setCurrentPage] = useState(1);
	const [currentRows, setCurrentRows] = useState(rows.slice(0, 10));
	const [totalPages, setTotalPages] = useState(Math.ceil(rows.length / 10))
	return (
		<>
			<table className="table table-hover table-responsive nowrap">
            <thead>
                <th>#</th>
                {
                    columns.map(column => <th key={column.label}>{column.label}</th>)
                }
            </thead>
            <tbody>
                {
                    currentRows.map((row, index) => {
                        return <tr key={index}>
                            <td><strong>{index + 1}</strong></td>
                            {
                                columns.map(column => {
                                    let count = dataCount.current++;
                                    return <td key={`${column.label}${index}`} 
                                        ref={el => refs.current[count] = el} 
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
        	<nav aria-label="Page navigation example">
			  <ul class="pagination pg-blue pagination-sm">
			  	{
			  		range(0, totalPages).map(page => <li className="page-item disabled={(page + 1) === currentPage}">
			  			<a className="page-link">
			  				{page + 1}
			  			</a>
			  		</li>)
			  	}
			  </ul>
			</nav>
		</>
	);
};

export default Table;