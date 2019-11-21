import React from 'react';
import CSRFToken from '../../../utility/CSRFToken';
import {onChangeHandler, sendFile} from "../../../store/actions/fileUpload";
import { connect } from 'react-redux';
import { MDBInputGroup, MDBBtn, MDBIcon } from "mdbreact";


const Form = ({file, changeHandler, onSendFileHandler}) => {
    return (
        <form className="col-md-5 mt-5" method="POST" encType="multipart/form-data">
            <CSRFToken/>
            
        <MDBInputGroup
          prepend={
            <MDBBtn
              color="default"
              className="m-0 px-3 py-2 z-depth-0"
              onClick={() => onSendFileHandler(file)}
            >
              <MDBIcon icon={"paper-plane"} />
            </MDBBtn>
          }
          inputs={
            <div className="custom-file">
                <input
                    type="file"
                    id="inputGroupFile01"
                    aria-describedby="inputGroupFileAddon01"
                    onChange={changeHandler}
                />
                <label className="custom-file-label" htmlFor="inputGroupFile01">
                    Parcourir
                </label>
            </div>
          }
          containerClassName="mb-3"
        />
            {/* <div className="input-group ml-2">
                
                <a type="submit" className="btn btn-success mb-2" onClick={() => onSendFileHandler(file)}>Charger</a>
            </div> */}
        </form>
    )
};
const mapStateToProps = state => {
    return {
        file: state.fileUpload.file,
    }
};

const mapDispatchToProps = dispatch => {
    return {
        changeHandler: event => dispatch(onChangeHandler(event)),
        onSendFileHandler: file => dispatch(sendFile(file))
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(Form);
