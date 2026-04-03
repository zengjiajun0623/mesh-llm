// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {MerkleProof} from "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

/// @title MeshTokenTestnet — Same as MeshToken but with 5-minute epochs for testing
contract MeshTokenTestnet is ERC20 {
    uint256 public constant MAX_SUPPLY = 21_000_000e8;
    uint256 public constant INITIAL_REWARD = 7_192e8;
    uint256 public constant HALVING_INTERVAL = 1_460;
    uint256 public constant EPOCH_DURATION = 5 minutes;        // 5 min for testnet
    uint256 public constant SUBMISSION_WINDOW = 5 minutes;     // 5 min for testnet
    uint256 public constant CLAIM_EXPIRY = 365 days;
    uint256 public constant FREE_TIER_WEIGHT = 2_000;

    uint256 public totalMinted;
    uint256 public immutable genesisTimestamp;

    mapping(uint256 => mapping(address => uint256)) public epochWork;
    mapping(uint256 => uint256) public epochTotalWork;
    mapping(uint256 => uint256) public epochNodeCount;
    mapping(uint256 => mapping(address => bool)) public claimed;
    mapping(uint256 => mapping(address => bytes32)) public workProofRoots;

    event WorkSubmitted(uint256 indexed epoch, address indexed node, uint256 effectiveWork, bytes32 proofRoot);
    event Claimed(uint256 indexed epoch, address indexed node, uint256 amount);

    constructor() ERC20("Mesh", "MESH") {
        genesisTimestamp = block.timestamp;
    }

    function decimals() public pure override returns (uint8) { return 8; }

    function currentEpoch() public view returns (uint256) {
        return (block.timestamp - genesisTimestamp) / EPOCH_DURATION;
    }

    function epochEnd(uint256 epoch) public view returns (uint256) {
        return genesisTimestamp + (epoch + 1) * EPOCH_DURATION;
    }

    function rewardForEpoch(uint256 epoch) public pure returns (uint256) {
        uint256 era = epoch / HALVING_INTERVAL;
        if (era >= 64) return 0;
        return INITIAL_REWARD >> era;
    }

    function maxShareBps(uint256 epoch) public view returns (uint256) {
        uint256 n = epochNodeCount[epoch];
        if (n == 0) return 10_000;
        uint256 fair = 10_000 / n;
        return fair > 1_000 ? fair : 1_000;
    }

    function submitWork(uint256 epoch, uint256 paidGpuSeconds, uint256 freeGpuSeconds, bytes32 consumerProofRoot) external {
        require(epoch < currentEpoch(), "epoch not closed");
        require(block.timestamp <= epochEnd(epoch) + SUBMISSION_WINDOW, "submission window closed");
        require(workProofRoots[epoch][msg.sender] == bytes32(0), "already submitted");
        require(paidGpuSeconds > 0 || freeGpuSeconds > 0, "no work");

        uint256 effectiveWork = paidGpuSeconds + (freeGpuSeconds * FREE_TIER_WEIGHT) / 10_000;

        workProofRoots[epoch][msg.sender] = consumerProofRoot;
        epochWork[epoch][msg.sender] = effectiveWork;
        epochTotalWork[epoch] += effectiveWork;
        epochNodeCount[epoch] += 1;

        emit WorkSubmitted(epoch, msg.sender, effectiveWork, consumerProofRoot);
    }

    function claim(uint256 epoch) public {
        require(block.timestamp > epochEnd(epoch) + SUBMISSION_WINDOW, "submission window open");
        require(block.timestamp <= epochEnd(epoch) + CLAIM_EXPIRY, "claim expired");
        require(!claimed[epoch][msg.sender], "already claimed");
        require(epochWork[epoch][msg.sender] > 0, "no work");

        claimed[epoch][msg.sender] = true;

        uint256 reward = rewardForEpoch(epoch);
        uint256 remaining = MAX_SUPPLY - totalMinted;
        if (reward > remaining) reward = remaining;
        if (reward == 0) return;

        uint256 nodeWork = epochWork[epoch][msg.sender];
        uint256 totalWork = epochTotalWork[epoch];
        uint256 share = (reward * nodeWork) / totalWork;

        uint256 cap = (reward * maxShareBps(epoch)) / 10_000;
        if (share > cap) share = cap;

        totalMinted += share;
        _mint(msg.sender, share);

        emit Claimed(epoch, msg.sender, share);
    }

    function claimBatch(uint256[] calldata epochs) external {
        for (uint256 i = 0; i < epochs.length; i++) {
            claim(epochs[i]);
        }
    }
}
