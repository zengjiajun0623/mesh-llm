// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {MeshToken} from "../src/MeshToken.sol";
import {PaymentChannel} from "../src/PaymentChannel.sol";

contract PaymentChannelTest is Test {
    MeshToken public token;
    PaymentChannel public channel;

    uint256 consumerKey = 0xA11CE;
    address consumer = vm.addr(consumerKey);
    address provider = makeAddr("provider");

    function setUp() public {
        token = new MeshToken();
        channel = new PaymentChannel(address(token));

        // Give consumer some MESH by simulating mining
        // Advance epoch, submit work, claim
        vm.warp(block.timestamp + 1 days + 1);
        vm.prank(consumer);
        token.submitWork(0, 10_000, 0, keccak256("proof"));
        vm.warp(token.epochEnd(0) + token.SUBMISSION_WINDOW() + 1);
        vm.prank(consumer);
        token.claim(0);

        // Consumer approves channel contract
        vm.prank(consumer);
        token.approve(address(channel), type(uint256).max);
    }

    function _signState(uint256 channelId, uint256 amount, uint256 nonce) internal view returns (bytes memory) {
        bytes32 stateHash = keccak256(abi.encode(channelId, amount, nonce));
        bytes32 ethHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", stateHash));
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(consumerKey, ethHash);
        return abi.encodePacked(r, s, v);
    }

    // ── Open ──────────────────────────────────────────────────────────

    function test_open() public {
        uint256 balBefore = token.balanceOf(consumer);

        vm.prank(consumer);
        uint256 id = channel.open(provider, 1000e8, 7 days);

        assertEq(id, 0);
        assertEq(token.balanceOf(consumer), balBefore - 1000e8);
        assertEq(token.balanceOf(address(channel)), 1000e8);
        assertTrue(channel.isOpen(id));
        assertEq(channel.channelBalance(id), 1000e8);
    }

    function test_open_revert_selfChannel() public {
        vm.prank(consumer);
        vm.expectRevert("self channel");
        channel.open(consumer, 1000e8, 7 days);
    }

    function test_open_revert_zeroDeposit() public {
        vm.prank(consumer);
        vm.expectRevert("zero deposit");
        channel.open(provider, 0, 7 days);
    }

    // ── Top up ────────────────────────────────────────────────────────

    function test_topUp() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 500e8, 7 days);

        vm.prank(consumer);
        channel.topUp(id, 300e8);

        assertEq(channel.channelBalance(id), 800e8);
    }

    // ── Settle ────────────────────────────────────────────────────────

    function test_settle() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 1000e8, 7 days);

        bytes memory sig = _signState(id, 100e8, 1);

        vm.prank(provider);
        channel.settle(id, 100e8, 1, sig);

        assertEq(token.balanceOf(provider), 100e8);
        assertEq(channel.channelBalance(id), 900e8);
    }

    function test_settle_incremental() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 1000e8, 7 days);

        // First settlement
        bytes memory sig1 = _signState(id, 100e8, 1);
        vm.prank(provider);
        channel.settle(id, 100e8, 1, sig1);

        // Second settlement (cumulative)
        bytes memory sig2 = _signState(id, 350e8, 2);
        vm.prank(provider);
        channel.settle(id, 350e8, 2, sig2);

        assertEq(token.balanceOf(provider), 350e8);
        assertEq(channel.channelBalance(id), 650e8);
    }

    function test_settle_revert_notProvider() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 1000e8, 7 days);

        bytes memory sig = _signState(id, 100e8, 1);

        vm.prank(consumer);
        vm.expectRevert("not provider");
        channel.settle(id, 100e8, 1, sig);
    }

    function test_settle_revert_exceedsDeposit() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 100e8, 7 days);

        bytes memory sig = _signState(id, 200e8, 1);

        vm.prank(provider);
        vm.expectRevert("exceeds deposit");
        channel.settle(id, 200e8, 1, sig);
    }

    function test_settle_revert_staleNonce() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 1000e8, 7 days);

        bytes memory sig1 = _signState(id, 100e8, 1);
        vm.prank(provider);
        channel.settle(id, 100e8, 1, sig1);

        // Try settling with same nonce
        bytes memory sig2 = _signState(id, 200e8, 1);
        vm.prank(provider);
        vm.expectRevert("stale nonce");
        channel.settle(id, 200e8, 1, sig2);
    }

    // ── Expire ────────────────────────────────────────────────────────

    function test_expire() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 1000e8, 7 days);

        // Settle some first
        bytes memory sig = _signState(id, 300e8, 1);
        vm.prank(provider);
        channel.settle(id, 300e8, 1, sig);

        // Advance past expiry
        vm.warp(block.timestamp + 7 days);

        uint256 balBefore = token.balanceOf(consumer);
        vm.prank(consumer);
        channel.expire(id);

        assertFalse(channel.isOpen(id));
        assertEq(token.balanceOf(consumer), balBefore + 700e8); // refund
    }

    function test_expire_revert_notExpired() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 1000e8, 7 days);

        vm.prank(consumer);
        vm.expectRevert("not expired");
        channel.expire(id);
    }

    // ── Close by provider ─────────────────────────────────────────────

    function test_closeByProvider() public {
        vm.prank(consumer);
        uint256 id = channel.open(provider, 1000e8, 7 days);

        bytes memory sig = _signState(id, 600e8, 1);

        uint256 consumerBal = token.balanceOf(consumer);
        vm.prank(provider);
        channel.closeByProvider(id, 600e8, 1, sig);

        assertFalse(channel.isOpen(id));
        assertEq(token.balanceOf(provider), 600e8);
        assertEq(token.balanceOf(consumer), consumerBal + 400e8); // refund
    }
}
