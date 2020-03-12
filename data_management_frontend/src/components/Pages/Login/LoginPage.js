import React, {useState} from "react";
import {MDBContainer, MDBBtn, MDBInput} from 'mdbreact';
import CSRFToken from '../../../utility/CSRFToken.js';
import Spinner from '../../UI/Spinner/Spinner.js';
import {connect} from 'react-redux';
import { authLogin, register } from '../../../store/actions/';
import Grid from "../../UI/Grid/Grid";

const LoginPage = props => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');


    const handleChange = event => {
        event.preventDefault();
        const value = event.target.value;
        switch (event.target.id) {
            case 'username':
                setUsername(value);
                break;
            case 'password':
                setPassword(value);
                break;
        }
    };

    const handleSubmit = event => {
        event.preventDefault();
        props.onAuth(username, password);
    };
    let errorMessage = null;

    if (props.error) {
        errorMessage = <div className={"alert alert-danger mt-5"} role={"alert"}>
             Nom d'utilisateur ou mot de passe incorrect.
            </div>;
    }
    return (
        <MDBContainer>
            <Grid>
                {errorMessage}
            </Grid>
            {
                (!props.isAuthenticated && props.loading) ? <Grid><Spinner/></Grid> : <Grid>
                    <form className="mt-5" method="POST">
                        <CSRFToken/>
                        <p className="h5 text-center mb-4">Sign In</p>
                        <div className="grey-text">
                        <MDBInput
                            label="Username"
                            icon="envelope"
                            group
                            type="text"
                            id="username"
                            validate
                            error="wrong"
                            success="right"
                            onChange={handleChange}
                        />
                        <MDBInput
                            label="Password"
                            icon="lock"
                            id="password"
                            group
                            type="password"
                            validate
                            onChange={handleChange}
                        />
                        </div>
                        <div className="text-center">
                            <MDBBtn onClick={handleSubmit}>Sign in</MDBBtn>
                            <MDBBtn onClick={props.onRegister} color="primary">Sign Up</MDBBtn>
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
        error: state.auth.error
    }
};

const mapDispatchToProps = dispatch => {
    return {
        onAuth: (username, password) => dispatch(authLogin(username, password)),
        onRegister: () => dispatch(register()),
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(LoginPage);