// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {MeshToken} from "../src/MeshToken.sol";
import {MeshPaymaster} from "../src/MeshPaymaster.sol";

contract MeshPaymasterTest is Test {
    MeshToken public token;
    MeshPaymaster public paymaster;

    address alice = makeAddr("alice");
    address owner;

    receive() external payable {}

    function setUp() public {
        owner = address(this);
        token = new MeshToken();
        // meshPerGas: 1e8 means 1 MESH (in 8-decimal) per 1e18 wei of gas cost
        // So for 100k gas at 10 gwei = 1e15 wei cost → meshCost = 1e15 * 1e8 * 11000 / (1e18 * 10000) = 110e3 = 0.0011 MESH
        paymaster = new MeshPaymaster(address(token), 1e8);

        vm.deal(address(paymaster), 10 ether);

        // Give alice some MESH by mining
        vm.warp(block.timestamp + 1 days + 1);
        vm.prank(alice);
        token.submitWork(0, 10_000, 0, keccak256("proof"));
        vm.warp(token.epochEnd(0) + token.SUBMISSION_WINDOW() + 1);
        vm.prank(alice);
        token.claim(0);

        vm.prank(alice);
        token.approve(address(paymaster), type(uint256).max);
    }

    function test_estimateMeshCost() public view {
        // 100_000 gas at 10 gwei = 1e15 wei
        // meshCost = (1e15 * 1e8 * 11000) / (1e18 * 10000) = 110_000 (0.0011 MESH in 8 dec)
        uint256 cost = paymaster.estimateMeshCost(100_000, 10 gwei);
        assertEq(cost, 110_000);
    }

    function test_validateSponsor() public view {
        bool ok = paymaster.validateSponsor(alice, 100_000, 10 gwei);
        assertTrue(ok);
    }

    function test_validateSponsor_noEth() public {
        paymaster.withdraw(owner, 10 ether, 0);
        bool ok = paymaster.validateSponsor(alice, 100_000, 10 gwei);
        assertFalse(ok);
    }

    function test_chargeMeshFee() public {
        uint256 balBefore = token.balanceOf(alice);
        paymaster.chargeMeshFee(alice, 100_000, 10 gwei);
        uint256 fee = paymaster.estimateMeshCost(100_000, 10 gwei);
        assertEq(token.balanceOf(alice), balBefore - fee);
        assertEq(token.balanceOf(address(paymaster)), fee);
    }

    function test_setMeshPerGas() public {
        paymaster.setMeshPerGas(2e8);
        assertEq(paymaster.meshPerGas(), 2e8);
    }

    function test_setFeeMarkup() public {
        paymaster.setFeeMarkup(15_000);
        assertEq(paymaster.feeMarkup(), 15_000);
    }

    function test_setFeeMarkup_revert_tooLow() public {
        vm.expectRevert("markup below 1x");
        paymaster.setFeeMarkup(9_000);
    }

    function test_setFeeMarkup_revert_tooHigh() public {
        vm.expectRevert("markup above 2x");
        paymaster.setFeeMarkup(21_000);
    }

    function test_withdraw() public {
        uint256 ethBefore = owner.balance;
        paymaster.withdraw(owner, 1 ether, 0);
        assertEq(owner.balance, ethBefore + 1 ether);
    }

    function test_receiveEth() public {
        uint256 balBefore = address(paymaster).balance;
        (bool ok,) = address(paymaster).call{value: 1 ether}("");
        assertTrue(ok);
        assertEq(address(paymaster).balance, balBefore + 1 ether);
    }
}
