import React from 'react';
import { MDBNavbar, MDBNavbarBrand, MDBNavbarNav, MDBNavItem, MDBDropdown,
MDBDropdownToggle, MDBDropdownMenu, MDBDropdownItem, MDBIcon } from "mdbreact"
import { connect } from 'react-redux';
import * as actions from '../store/actions/auth';


const Header = props => {
	return (
		<MDBNavbar>
			<MDBNavbarBrand>
				<strong>Data Management</strong>
			</MDBNavbarBrand>
			{
				props.isAuthenticated ? <MDBNavbarNav left>
									<MDBNavItem>
						              <MDBDropdown>
						                <MDBDropdownToggle nav caret>
						                  <MDBIcon icon="user" />
						                </MDBDropdownToggle>
						                <MDBDropdownMenu className="dropdown-default">
						                  <MDBDropdownItem href="#!" onClick={props.logout}>Déconnexion</MDBDropdownItem>
						                </MDBDropdownMenu>
						              </MDBDropdown>
						            </MDBNavItem>
								</MDBNavbarNav> : null
			}
		</MDBNavbar>
	);
}

const mapStateToProps = state => {
	return {
		isAuthenticated: state.token !== null,
	}
};

const mapDispatchToProps = dispatch => {
  return {
    logout: () => dispatch(actions.logout())
  } 
}

export default connect(mapStateToProps, mapDispatchToProps)(Header);