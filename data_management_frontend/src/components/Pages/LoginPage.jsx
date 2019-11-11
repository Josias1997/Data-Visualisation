import React, {useState} from "react";
import { MDBContainer, MDBRow, MDBCol, MDBBtn } from 'mdbreact';
import CSRFToken from '../../utils/CSRFToken.jsx';
import Spinner from '../UI/Spinner.jsx';
import { connect } from 'react-redux';
import * as actions from '../../store/actions/auth';

const LoginPage = props => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');


  const handleChange = event => {
    event.preventDefault();
    const value = event.target.value;
    switch(event.target.id) {
      case 'username': 
        setUsername(value);
        break;
      case 'password': 
        setPassword(value);
        break;
    }
  }

  const handleSubmit = event => {
    event.preventDefault();
    props.onAuth(username, password);
  } 
  let errorMessage = null;

  if (props.error) {
    errorMessage = <strong>Nom d'utilisateur ou mot de passe incorrect.</strong>;
  }
  return (
    <MDBContainer>
      {errorMessage}
    {
      (!props.isAuthenticated && props.loading) ? <Spinner /> : <MDBRow>
        <MDBCol md="3"></MDBCol>
        <MDBCol md="6">
          <form className="mt-3" onSubmit={handleSubmit} method="POST">
          <CSRFToken />
            <p className="h4 text-center mb-4">Connexion</p>
            <label htmlFor="username" className="grey-text">
              Pseudonyme
            </label>
            <input
              type="text"
              id="username"
              className="form-control"
              onChange={handleChange}
            />
            <br />
            <label htmlFor="password" className="grey-text">
              Mot de passe
            </label>
            <input
              type="password"
              id="password"
              className="form-control"
              onChange={handleChange}
            />
              <div className="text-center mt-4">
                  <MDBBtn color="indigo" type="submit">Valider</MDBBtn>
              </div>
          </form>
        </MDBCol>
         <MDBCol md="3"></MDBCol>
      </MDBRow>
    }
    </MDBContainer>
  );
};

const mapStateToProps = state => {
  return {
    isAuthenticated: state.token !== null,
    loading: state.loading,
    error: state.error
  }
};

const mapDispatchToProps = dispatch => {
  return {
    onAuth: (username, password) => dispatch(actions.authLogin(username, password))
  } 
}

export default connect(mapStateToProps, mapDispatchToProps)(LoginPage);