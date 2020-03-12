import React, {useState} from "react";
import { MDBContainer, MDBRow, MDBCol, MDBBtn } from 'mdbreact';
import CSRFToken from '../../../utility/CSRFToken.js';
import Spinner from '../../UI/Spinner/Spinner.js';
import { connect } from 'react-redux';
import { authSignup, login } from '../../../store/actions/';
import Grid from "../../UI/Grid/Grid";

const SignUpPage = props => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [validationError, setValidationError] = useState('');

  const handleChange = event => {
    event.preventDefault();
    const value = event.target.value;
    switch(event.target.id) {
      case 'username': 
        setUsername(value);
        break;
      case 'email': 
        setEmail(value);
        break;
      case 'password': 
        setPassword(value);
        break;
    }
  }

  const handleSubmit = event => {
    event.preventDefault();
    props.onSignUp(username, email, password);
  }
  let errorMessage = null;

  if (props.error) {
    errorMessage = <div className={"alert alert-danger mt-5"} role={"alert"}>
          Username already exists
        </div>;
  }
  return (
    <MDBContainer>
    {errorMessage}
    {
      (!props.isAuthenticated && props.loading) ? <Grid><Spinner /></Grid> : <Grid>
          <form className="mt-3" onSubmit={handleSubmit}>
            <CSRFToken />
            <p className="h4 text-center mb-4">Sign Up</p>
            <label htmlFor="username" className="grey-text">
              Username
            </label>
            <input
              type="text"
              id="username"
              className="form-control"
              onChange={handleChange}
            />
            <br />
            <label htmlFor="email" className="grey-text">
              Email
            </label>
            <input
              type="email"
              id="email"
              className="form-control"
              onChange={handleChange}
            />
            <br />
            <label htmlFor="password" className="grey-text">
              Password
            </label>
            <input
              type="password"
              id="password"
              className="form-control"
              onChange={handleChange}
            />
            <div className="row d-flex justify-content-center">
              <div className="text-center mt-4">
                  <MDBBtn onClick={handleSubmit}>Sign Up</MDBBtn>
              </div>
              <div className="text-center mt-4" onClick={props.onLogin}>
                  <MDBBtn color="primary">Sign In</MDBBtn>
              </div>
            </div>
          </form>
        </Grid>
    }
    </MDBContainer>
  );
};

const mapStateToProps = state => {
  return {
    isAuthenticated: state.auth.token !== null,
    loading: state.auth.loading,
    error: state.auth.error,
  }
};

const mapDispatchToProps = dispatch => {
  return {
    onSignUp: (username, email, password) => dispatch(authSignup(username, email, password)),
    onLogin: () => dispatch(login())
  } 
}

export default connect(mapStateToProps, mapDispatchToProps)(SignUpPage);
