import React, {useState} from 'react';
import {
    MDBNavbar, MDBNavbarBrand, MDBNavbarNav, MDBNavItem, MDBDropdown,
    MDBDropdownToggle, MDBDropdownMenu, MDBDropdownItem, MDBIcon, MDBNavbarToggler,
    MDBCollapse, MDBNavLink
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
                            <MDBNavLink to={"/"}>Manipulation</MDBNavLink>
                        </MDBNavItem>
                        <MDBNavItem>
                            <MDBNavLink to={"/r-statistics"}>Statistiques</MDBNavLink>
                        </MDBNavItem>
                        <MDBNavItem>
                            <MDBNavLink to={"/modelisation"}>Modélisation</MDBNavLink>
                        </MDBNavItem>
                        <MDBNavItem>
                            <MDBDropdown>
                                <MDBDropdownToggle nav caret>
                                    <MDBIcon icon="power-off"/>
                                </MDBDropdownToggle>
                                <MDBDropdownMenu className="dropdown-default">
                                    <MDBDropdownItem href="#!" onClick={props.logout}>Déconnexion</MDBDropdownItem>
                                </MDBDropdownMenu>
                            </MDBDropdown>
                        </MDBNavItem>
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