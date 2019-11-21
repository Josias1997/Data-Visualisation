/* Amélioration : Ajout Inscription d'utlisateur */

/*import React, {useState} from "react";
import { MDBContainer, MDBRow, MDBCol, MDBBtn } from 'mdbreact';
import CSRFToken from '../../utility/CSRFToken.js';
import Spinner from '../UI/Spinner.js';
import { connect } from 'react-redux';
import * as actions from '../../store/actions/auth';

const SignUpPage = props => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirmed, setPasswordConfirmed] = useState('');

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
      case 'passwordConfirmed': 
        setPasswordConfirmed(value);
        break;
    }
  }

  const handleSubmit = event => {
    event.preventDefault();
    props.onSignUp(username, email, password, passwordConfirmed);
  }
  let errorMessage = null;

  if (props.error) {
    errorMessage = <p>{props.error.message}</p>
  }
  return (
    <MDBContainer>
    {errorMessage}
    {
      (!props.isAuthenticated && props.loading) ? <Spinner /> : <MDBRow>
        <MDBCol md="3"></MDBCol>
        <MDBCol md="6">
          <form className="mt-3" onSubmit={handleSubmit}>
            <CSRFToken />
            <p className="h4 text-center mb-4">Inscription</p>
            <label htmlFor="username" className="grey-text">
              Nom d'utilisateur
            </label>
            <input
              type="text"
              id="username"
              className="form-control"
            />
            <br />
            <label htmlFor="email" className="grey-text">
              Email
            </label>
            <input
              type="email"
              id="email"
              className="form-control"
            />
            <br />
            <label htmlFor="password" className="grey-text">
              Mot de passe
            </label>
            <input
              type="password"
              id="password"
              className="form-control"
            />
            <label htmlFor="passwordConfirmed" className="grey-text">
              Confirmer Mot de passe
            </label>
            <input
              type="password"
              id="passwordConfirmed"
              className="form-control"
            />
            <div className="row d-flex justify-content-center">
              <div className="text-center mt-4">
                  <MDBBtn color="indigo" type="submit">Valider</MDBBtn>
              </div>
              <div className="text-center mt-4" onClick={props.toggle}>
                  <MDBBtn color="green" type="submit">Se connecter</MDBBtn>
              </div>
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
    error: state.error,
  }
};

const mapDispatchToProps = dispatch => {
  return {
    onSignUp: (username, email, password, passwordConfirmed) => dispatch(actions.authSignup(username, email, password, passwordConfirmed))
  } 
}

export default connect(mapStateToProps, mapDispatchToProps)(SignUpPage);
*/