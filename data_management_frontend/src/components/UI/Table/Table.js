import React, {useRef, useState} from "react";
import { range } from "../../../utility/utility";


const Table = ({ columns, rows }) => {
	const dataCount = useRef(0);
	const refs = useRef([]);
	const [numberRowsPerPage, setNumberRowsPerPage] = useState(10);
	const [currentPage, setCurrentPage] = useState(1);
	const [currentRows, setCurrentRows] = useState(rows.slice(0, 10));
	const [totalPages, setTotalPages] = useState(Math.ceil(rows.length / 10))

    const changePage = (page) => {
        let add = 0;
        if (page > 1) {
            add = 1;
        }
        setCurrentRows(rows.slice(numberRowsPerPage * (page - 1) + add, numberRowsPerPage * page));
        setCurrentPage(page);
    };
	return (
		<>
			<table className="table table-hover table-responsive nowrap">
                <thead>
                    <tr>
                    <th>#</th>
                        {
                            columns.map(column => <th key={column.field}>{column.field}</th>)
                        }
                    </tr>
                </thead>
                <tbody>
                    {
                        currentRows.map((row, index) => {
                            return <tr key={index}>
                                <td><strong>{index + 1}</strong></td>
                                {
                                    columns.map(column => {
                                        let count = dataCount.current++;
                                        return <td key={`${column.field}${index}`} 
                                            ref={el => refs.current[count] = el} 
                                            style={{
                                                cursor: 'pointer'
                                            }}
                                        >
                                            {row[column.field]}
                                        </td>
                                    })
                                }
                            </tr>
                        })
                    }
                </tbody>
        	</table>
        	<nav>
			  <ul className="pagination pg-blue pagination-sm">
			  	{
			  		range(0, totalPages).map(page => <li key={page} className="page-item" disabled>
			  			<a className="page-link" onClick={() => changePage(page + 1)}>
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