import React from "react";

const AlignCenter = props => {
    let style = undefined;
    if(props.style) {
        style = props.style;
    }
    return (
        <div className={"row justify-content-center"} style={style}>
            {props.children}
        </div>
    );
}

export default AlignCenter