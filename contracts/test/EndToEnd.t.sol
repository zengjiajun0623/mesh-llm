// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {MeshToken} from "../src/MeshToken.sol";
import {PaymentChannel} from "../src/PaymentChannel.sol";
import {MeshPaymaster} from "../src/MeshPaymaster.sol";

/// @notice End-to-end test: user joins mesh, mines, checks balance, claims,
///         opens payment channel, pays for inference, provider settles.
contract EndToEndTest is Test {
    MeshToken public token;
    PaymentChannel public channel;
    MeshPaymaster public paymaster;

    // Alice: MacBook miner serving Qwen3-4B (free tier mostly)
    uint256 aliceKey = 0xA11CE;
    address alice = vm.addr(aliceKey);

    // Bob: GPU workstation serving DeepSeek-R1-32B (paid tier)
    uint256 bobKey = 0xB0B;
    address bob = vm.addr(bobKey);

    // Carol: consumer agent wanting inference
    uint256 carolKey = 0xCA201;
    address carol = vm.addr(carolKey);

    function setUp() public {
        // Deploy contracts
        token = new MeshToken();
        channel = new PaymentChannel(address(token));
        paymaster = new MeshPaymaster(address(token), 1e8);
        vm.deal(address(paymaster), 100 ether);
    }

    function _signChannelState(uint256 key, uint256 channelId, uint256 amount, uint256 nonce) internal pure returns (bytes memory) {
        bytes32 stateHash = keccak256(abi.encode(channelId, amount, nonce));
        bytes32 ethHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", stateHash));
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(key, ethHash);
        return abi.encodePacked(r, s, v);
    }

    function test_fullMiningCycle() public {
        console.log("=== Day 1: Alice and Bob join the mesh and serve inference ===");

        // Advance past epoch 0
        vm.warp(token.genesisTimestamp() + 1 days + 1);

        // Alice served 200 GPU-seconds (all free tier)
        vm.prank(alice);
        token.submitWork(0, 0, 200_000, keccak256("alice_epoch0"));
        // effective: 200_000 * 2000 / 10000 = 40_000

        // Bob served 800 GPU-seconds (all paid tier)
        vm.prank(bob);
        token.submitWork(0, 800_000, 0, keccak256("bob_epoch0"));
        // effective: 800_000

        console.log("Alice effective work:", token.epochWork(0, alice));
        console.log("Bob effective work:", token.epochWork(0, bob));
        console.log("Total network work:", token.epochTotalWork(0));

        console.log("");
        console.log("=== Day 4: Submission window closes, miners can claim ===");

        vm.warp(token.epochEnd(0) + token.SUBMISSION_WINDOW() + 1);

        // Check claimable amounts before claiming
        uint256 reward = token.rewardForEpoch(0);
        uint256 aliceWork = token.epochWork(0, alice);
        uint256 bobWork = token.epochWork(0, bob);
        uint256 totalWork = token.epochTotalWork(0);

        uint256 aliceRawShare = (reward * aliceWork) / totalWork;
        uint256 bobRawShare = (reward * bobWork) / totalWork;

        console.log("Epoch 0 reward:", reward);
        console.log("Alice raw share:", aliceRawShare);
        console.log("Bob raw share:", bobRawShare);

        // Alice claims
        vm.prank(alice);
        token.claim(0);
        uint256 aliceBalance = token.balanceOf(alice);
        console.log("Alice balance after claim:", aliceBalance);
        assertGt(aliceBalance, 0, "Alice should have mined tokens");

        // Bob claims
        vm.prank(bob);
        token.claim(0);
        uint256 bobBalance = token.balanceOf(bob);
        console.log("Bob balance after claim:", bobBalance);
        assertGt(bobBalance, 0, "Bob should have mined tokens");

        // Bob should earn much more (paid tier = 1.0x vs free tier = 0.2x)
        assertGt(bobBalance, aliceBalance * 10, "Bob (paid) should earn >10x Alice (free)");

        console.log("");
        console.log("=== Alice opens payment channel to use Bob's model ===");

        // Alice approves and opens channel to Bob
        vm.prank(alice);
        token.approve(address(channel), type(uint256).max);

        uint256 deposit = aliceBalance / 2; // deposit half her balance
        vm.prank(alice);
        uint256 channelId = channel.open(bob, deposit, 30 days);

        console.log("Channel opened, ID:", channelId);
        console.log("Deposit:", deposit);
        assertEq(channel.channelBalance(channelId), deposit);

        console.log("");
        console.log("=== Alice uses Bob's model, pays per request ===");

        // Simulate 5 inference requests, each costing ~1/10th of deposit
        uint256 costPerRequest = deposit / 10;
        uint256 cumulativePayment = 0;

        for (uint256 i = 1; i <= 5; i++) {
            cumulativePayment += costPerRequest;
            bytes memory sig = _signChannelState(aliceKey, channelId, cumulativePayment, i);

            vm.prank(bob);
            channel.settle(channelId, cumulativePayment, i, sig);
        }

        uint256 bobEarned = token.balanceOf(bob) - bobBalance; // earnings from channel
        console.log("Bob earned from channel:", bobEarned);
        assertEq(bobEarned, costPerRequest * 5, "Bob should earn 5 request payments");

        uint256 remaining = channel.channelBalance(channelId);
        console.log("Channel remaining balance:", remaining);
        assertEq(remaining, deposit - (costPerRequest * 5));

        console.log("");
        console.log("=== Day 5: Epoch 1 - Bob earns more from paid requests ===");

        // Advance to epoch 1 close
        vm.warp(token.genesisTimestamp() + 2 days + 1);

        // Bob served 500 GPU-seconds (paid, including Alice's requests)
        vm.prank(bob);
        token.submitWork(1, 500_000, 0, keccak256("bob_epoch1"));

        // Alice served 100 GPU-seconds (free tier)
        vm.prank(alice);
        token.submitWork(1, 0, 100_000, keccak256("alice_epoch1"));

        vm.warp(token.epochEnd(1) + token.SUBMISSION_WINDOW() + 1);

        uint256 bobBefore = token.balanceOf(bob);
        vm.prank(bob);
        token.claim(1);
        uint256 bobEpoch1 = token.balanceOf(bob) - bobBefore;
        console.log("Bob epoch 1 mining reward:", bobEpoch1);

        uint256 aliceBefore = token.balanceOf(alice);
        vm.prank(alice);
        token.claim(1);
        uint256 aliceEpoch1 = token.balanceOf(alice) - aliceBefore;
        console.log("Alice epoch 1 mining reward:", aliceEpoch1);

        console.log("");
        console.log("=== Final state ===");
        console.log("Alice total balance:", token.balanceOf(alice));
        console.log("Bob total balance:", token.balanceOf(bob));
        console.log("Total MESH minted:", token.totalMinted());
        console.log("Max supply:", token.MAX_SUPPLY());

        // Verify total minted is reasonable (2 epochs worth)
        assertLe(token.totalMinted(), token.INITIAL_REWARD() * 2, "Should not exceed 2 epochs of rewards");
    }

    function test_batchClaimMultipleEpochs() public {
        console.log("=== Simulate a week of mining, then batch claim ===");

        // Alice mines for 7 epochs
        for (uint256 e = 0; e < 7; e++) {
            vm.warp(token.genesisTimestamp() + (e + 1) * 1 days + 1);
            vm.prank(alice);
            token.submitWork(e, 500_000, 200_000, keccak256(abi.encode("alice", e)));
        }

        // Advance past epoch 6's submission window
        vm.warp(token.epochEnd(6) + token.SUBMISSION_WINDOW() + 1);

        // Batch claim all 7 epochs
        uint256[] memory epochs = new uint256[](7);
        for (uint256 i = 0; i < 7; i++) epochs[i] = i;

        vm.prank(alice);
        token.claimBatch(epochs);

        uint256 balance = token.balanceOf(alice);
        console.log("Alice balance after 7-epoch batch claim:", balance);
        assertEq(balance, token.INITIAL_REWARD() * 7, "Should be 7 full epoch rewards (sole miner)");
        console.log("Total minted:", token.totalMinted());
    }

    function test_paymasterFlow() public {
        console.log("=== Paymaster: mine then use paymaster for gas ===");

        // Alice mines epoch 0
        vm.warp(token.genesisTimestamp() + 1 days + 1);
        vm.prank(alice);
        token.submitWork(0, 10_000, 0, keccak256("proof"));

        vm.warp(token.epochEnd(0) + token.SUBMISSION_WINDOW() + 1);
        vm.prank(alice);
        token.claim(0);

        uint256 balance = token.balanceOf(alice);
        console.log("Alice MESH balance:", balance);

        // Alice approves paymaster
        vm.prank(alice);
        token.approve(address(paymaster), type(uint256).max);

        // Paymaster validates Alice is eligible
        bool eligible = paymaster.validateSponsor(alice, 200_000, 20 gwei);
        assertTrue(eligible, "Alice should be eligible for sponsorship");

        // Simulate paymaster charging fee
        uint256 fee = paymaster.estimateMeshCost(200_000, 20 gwei);
        console.log("Estimated MESH fee for 200k gas:", fee);

        paymaster.chargeMeshFee(alice, 200_000, 20 gwei);

        console.log("Alice balance after gas fee:", token.balanceOf(alice));
        assertEq(token.balanceOf(alice), balance - fee);
        console.log("Paymaster MESH balance:", token.balanceOf(address(paymaster)));
    }
}
