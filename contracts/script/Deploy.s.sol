// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script, console} from "forge-std/Script.sol";
import {MeshToken} from "../src/MeshToken.sol";
import {PaymentChannel} from "../src/PaymentChannel.sol";
import {MeshPaymaster} from "../src/MeshPaymaster.sol";

/// @notice Deploy all MESH contracts to Ethereum (mainnet or Sepolia).
/// Usage:
///   forge script script/Deploy.s.sol --rpc-url $RPC_URL --broadcast --private-key $PRIVATE_KEY
contract DeployScript is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerKey);

        console.log("Deployer:", deployer);
        console.log("Chain ID:", block.chainid);

        vm.startBroadcast(deployerKey);

        // 1. Deploy MeshToken
        MeshToken token = new MeshToken();
        console.log("MeshToken deployed at:", address(token));

        // 2. Deploy PaymentChannel
        PaymentChannel channel = new PaymentChannel(address(token));
        console.log("PaymentChannel deployed at:", address(channel));

        // 3. Deploy MeshPaymaster
        // meshPerGas: 1e8 = 1 MESH (8 decimals) per 1e18 wei of gas cost
        // This means ~0.001 MESH per typical tx at 10 gwei
        MeshPaymaster paymaster = new MeshPaymaster(address(token), 1e8);
        console.log("MeshPaymaster deployed at:", address(paymaster));

        // 4. Fund paymaster with some ETH for gas sponsorship
        uint256 paymasterFunding = 0.1 ether;
        if (deployer.balance > paymasterFunding) {
            (bool ok,) = address(paymaster).call{value: paymasterFunding}("");
            require(ok, "paymaster funding failed");
            console.log("Paymaster funded with 0.1 ETH");
        }

        vm.stopBroadcast();

        console.log("");
        console.log("=== Deployment Summary ===");
        console.log("MeshToken:      ", address(token));
        console.log("PaymentChannel: ", address(channel));
        console.log("MeshPaymaster:  ", address(paymaster));
        console.log("Genesis epoch:  ", token.genesisTimestamp());
        console.log("Initial reward: ", token.INITIAL_REWARD(), "(7,192 MESH/day)");
        console.log("Max supply:     ", token.MAX_SUPPLY(), "(21,000,000 MESH)");
    }
}
