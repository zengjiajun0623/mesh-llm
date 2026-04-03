// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {MeshToken} from "../src/MeshToken.sol";

contract MeshTokenTest is Test {
    MeshToken public token;

    uint256 nodeKey = 0x1234;
    address node = vm.addr(nodeKey);

    uint256 aliceKey = 0xA11CE;
    address alice = vm.addr(aliceKey);

    uint256 bobKey = 0xB0B;
    address bob = vm.addr(bobKey);

    function setUp() public { token = new MeshToken(); }

    function _sign(uint256 key, address nodeAddr, uint256 ts) internal pure returns (bytes memory) {
        bytes32 h = keccak256(abi.encode(nodeAddr, vm.addr(key), ts));
        bytes32 eh = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", h));
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(key, eh);
        return abi.encodePacked(r, s, v);
    }

    function _claim1(uint256 consumerKey, address nodeAddr, uint256 ts) internal {
        address[] memory c = new address[](1); c[0] = vm.addr(consumerKey);
        uint256[] memory t = new uint256[](1); t[0] = ts;
        bytes[] memory s = new bytes[](1); s[0] = _sign(consumerKey, nodeAddr, ts);
        vm.prank(nodeAddr);
        token.claim(c, t, s);
    }

    // ── Basic ─────────────────────────────────────────────────────────

    function test_claim_single() public {
        _claim1(aliceKey, node, block.timestamp);
        assertEq(token.balanceOf(node), 1e18);
    }

    function test_claim_two_consumers() public {
        address[] memory c = new address[](2); c[0] = alice; c[1] = bob;
        uint256[] memory t = new uint256[](2); t[0] = block.timestamp; t[1] = block.timestamp;
        bytes[] memory s = new bytes[](2);
        s[0] = _sign(aliceKey, node, block.timestamp);
        s[1] = _sign(bobKey, node, block.timestamp);
        vm.prank(node);
        token.claim(c, t, s);
        assertEq(token.balanceOf(node), 2e18);
    }

    // ── Anti-gaming ───────────────────────────────────────────────────

    function test_selfDeal_rejected() public {
        address[] memory c = new address[](1); c[0] = node;
        uint256[] memory t = new uint256[](1); t[0] = block.timestamp;
        bytes[] memory s = new bytes[](1); s[0] = _sign(nodeKey, node, block.timestamp);
        vm.prank(node);
        vm.expectRevert("no valid receipts");
        token.claim(c, t, s);
    }

    function test_doubleSpend_rejected() public {
        _claim1(aliceKey, node, block.timestamp);
        address[] memory c = new address[](1); c[0] = alice;
        uint256[] memory t = new uint256[](1); t[0] = block.timestamp;
        bytes[] memory s = new bytes[](1); s[0] = _sign(aliceKey, node, block.timestamp);
        vm.prank(node);
        vm.expectRevert("no valid receipts");
        token.claim(c, t, s);
    }

    function test_rateLimit_within_hour() public {
        _claim1(aliceKey, node, block.timestamp);
        vm.warp(block.timestamp + 30 minutes);
        address[] memory c = new address[](1); c[0] = alice;
        uint256[] memory t = new uint256[](1); t[0] = block.timestamp;
        bytes[] memory s = new bytes[](1); s[0] = _sign(aliceKey, node, block.timestamp);
        vm.prank(node);
        vm.expectRevert("no valid receipts");
        token.claim(c, t, s);
    }

    function test_rateLimit_after_hour() public {
        _claim1(aliceKey, node, block.timestamp);
        vm.warp(block.timestamp + 1 hours + 1);
        _claim1(aliceKey, node, block.timestamp);
        assertEq(token.balanceOf(node), 2e18);
    }

    function test_future_timestamp_rejected() public {
        address[] memory c = new address[](1); c[0] = alice;
        uint256[] memory t = new uint256[](1); t[0] = block.timestamp + 1 days;
        bytes[] memory s = new bytes[](1); s[0] = _sign(aliceKey, node, block.timestamp + 1 days);
        vm.prank(node);
        vm.expectRevert("no valid receipts");
        token.claim(c, t, s);
    }

    function test_expired_receipt_rejected() public {
        uint256 ts = block.timestamp;
        vm.warp(block.timestamp + 8 days);
        address[] memory c = new address[](1); c[0] = alice;
        uint256[] memory t = new uint256[](1); t[0] = ts;
        bytes[] memory s = new bytes[](1); s[0] = _sign(aliceKey, node, ts);
        vm.prank(node);
        vm.expectRevert("no valid receipts");
        token.claim(c, t, s);
    }

    function test_bad_signature_rejected() public {
        address[] memory c = new address[](1); c[0] = alice;
        uint256[] memory t = new uint256[](1); t[0] = block.timestamp;
        bytes[] memory s = new bytes[](1); s[0] = _sign(bobKey, node, block.timestamp); // wrong key
        vm.prank(node);
        vm.expectRevert("no valid receipts");
        token.claim(c, t, s);
    }

    // ── Properties ────────────────────────────────────────────────────

    function test_maxSupply() public view {
        assertEq(token.MAX_SUPPLY(), 21_000_000e18);
        assertEq(token.mintableSupply(), 21_000_000e18);
    }

    function test_metadata() public view {
        assertEq(token.name(), "Mesh");
        assertEq(token.symbol(), "MESH");
        assertEq(token.decimals(), 18);
    }

    function test_multipleNodes() public {
        address node2 = makeAddr("node2");
        _claim1(aliceKey, node, block.timestamp);
        uint256 ts2 = block.timestamp + 1 hours + 1;
        vm.warp(ts2);
        address[] memory c = new address[](1); c[0] = alice;
        uint256[] memory t = new uint256[](1); t[0] = ts2;
        bytes[] memory s = new bytes[](1); s[0] = _sign(aliceKey, node2, ts2);
        vm.prank(node2);
        token.claim(c, t, s);
        assertEq(token.balanceOf(node), 1e18);
        assertEq(token.balanceOf(node2), 1e18);
    }
}
