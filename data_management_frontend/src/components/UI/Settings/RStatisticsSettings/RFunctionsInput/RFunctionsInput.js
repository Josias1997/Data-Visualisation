import React from "react";
import { MDBContainer, MDBInputGroup, MDBDropdown, MDBDropdownToggle, MDBIcon, MDBDropdownMenu, MDBDropdownItem } from "mdbreact";

const RFunctionsInput = props => {
    return (
      <div className="col-md-5 mt-2">
        <MDBInputGroup
          containerClassName="mb-3"
          append={
            <MDBDropdown>
              <MDBDropdownToggle
                color="default"
                size="md"
                className="m-0 px-3 z-depth-0"
              >
                Type <MDBIcon icon="caret-down" className="ml-1" />
              </MDBDropdownToggle>
              <MDBDropdownMenu>
                <MDBDropdownItem>Action</MDBDropdownItem>
                <MDBDropdownItem>Another Action</MDBDropdownItem>
                <MDBDropdownItem>Something else here</MDBDropdownItem>
                <MDBDropdownItem divider />
                <MDBDropdownItem>Separated link</MDBDropdownItem>
              </MDBDropdownMenu>
            </MDBDropdown>
          }
        />
      </div>
    );
};

export default RFunctionsInput;