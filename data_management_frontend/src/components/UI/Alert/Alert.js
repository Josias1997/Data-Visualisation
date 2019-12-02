import React from "react";

const Alert = props => {
    return (
        <div className="alert alert-danger mt-5" role="alert">
            {props.children}
        </div>
    )
};

export default Alert;