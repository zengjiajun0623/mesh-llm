// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {MeshToken} from "./MeshToken.sol";
import {PaymentChannel} from "./PaymentChannel.sol";
import {MeshPaymaster} from "./MeshPaymaster.sol";

/// @title MeshFactory — Deploy all MESH contracts in one tx
/// @notice Deploys MeshToken, PaymentChannel, and MeshPaymaster.
///         Callable from an Elytro smart account via a single UserOp.
contract MeshFactory {
    event Deployed(address token, address channel, address paymaster);

    function deploy(uint256 meshPerGas) external returns (address tokenAddr, address channelAddr, address paymasterAddr) {
        MeshToken token = new MeshToken();
        PaymentChannel channel = new PaymentChannel(address(token));
        MeshPaymaster paymaster = new MeshPaymaster(address(token), meshPerGas);

        emit Deployed(address(token), address(channel), address(paymaster));
        return (address(token), address(channel), address(paymaster));
    }
}
