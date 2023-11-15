// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Base {
	// Store a dynamic byte array
	bytes public Data;

	// Set the dynamic byte array
	function setData(bytes calldata _data) public {
		Data = "12344";
        // solidity does not support comparison like 
        // Assert(Data == "12344"); 
	}

}