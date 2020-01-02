import React, {useState} from 'react';
import {
    MDBNavbar, MDBNavbarBrand, MDBNavbarNav, MDBNavItem, MDBDropdown,
    MDBDropdownToggle, MDBDropdownMenu, MDBDropdownItem, MDBIcon, MDBNavbarToggler,
    MDBCollapse, MDBNavLink, MDBBtn
} from "mdbreact"
import {connect} from 'react-redux';
import * as actions from '../../store/actions/auth';


const Header = props => {
    const [isOpen, setIsOpen] = useState(false);
    const toggleCollapse = () => {
        setIsOpen(!isOpen);
    };

    return (
        <MDBNavbar color={"default-color"} dark expand={"md"}>
            <MDBNavbarBrand>
                <strong className={"white-text"}>Data Management</strong>
            </MDBNavbarBrand>
            <MDBNavbarToggler onClick={toggleCollapse}/>
            {
                props.isAuthenticated ? <MDBCollapse isOpen={isOpen} navbar>
                    <MDBNavbarNav left>
                        <MDBNavItem>
                            <MDBNavLink to={"/"}>Data Pre-processing</MDBNavLink>
                        </MDBNavItem>
                        <MDBNavItem>
                            <MDBNavLink to={"/r-statistics"}>Data Exploratory Analysis</MDBNavLink>
                        </MDBNavItem>
                        <MDBNavItem>
                            <MDBNavLink to={"/modelisation"}>Ready for Modelling</MDBNavLink>
                        </MDBNavItem>
                        <MDBNavItem>
                            <MDBNavLink to={"/machine-learning"}>Machine Learning</MDBNavLink>
                        </MDBNavItem>
                        <MDBNavItem>
                            <MDBNavLink to={"/deep-learning"}>Deep Learning</MDBNavLink>
                        </MDBNavItem>
                        <MDBNavItem>
                            <MDBNavLink to={"/text-mining"}>Text Mining</MDBNavLink>
                        </MDBNavItem>

                        <MDBNavItem>
                            <MDBNavLink to={"/dashboard"}>Dashboard</MDBNavLink>
                        </MDBNavItem>
                    </MDBNavbarNav>
                    <MDBNavbarNav right>
                        <MDBBtn onClick={props.logout}>
                            <MDBIcon icon="power-off"/>
                        </MDBBtn>
                    </MDBNavbarNav>
                </MDBCollapse> : null
            }
        </MDBNavbar>
    );
};

const mapStateToProps = state => {
    return {
        isAuthenticated: state.auth.token !== null,
    }
};

const mapDispatchToProps = dispatch => {
    return {
        logout: () => dispatch(actions.logout())
    }
};

export default connect(mapStateToProps, mapDispatchToProps)(Header);