// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {MerkleProof} from "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

/// @title MeshToken — Bitcoin-style issuance via Proof-of-Useful-Work
/// @notice ERC-20 with 21M fixed supply, daily epochs, halving every ~4 years.
///         Mined by serving AI inference on the mesh-llm network.
contract MeshToken is ERC20 {
    // ── Constants ──────────────────────────────────────────────────────
    uint256 public constant MAX_SUPPLY = 21_000_000e8;        // 8 decimals
    uint256 public constant INITIAL_REWARD = 7_192e8;          // 7,192 MESH per epoch
    uint256 public constant HALVING_INTERVAL = 1_460;          // ~4 years of daily epochs
    uint256 public constant EPOCH_DURATION = 1 days;
    uint256 public constant SUBMISSION_WINDOW = 3 days;
    uint256 public constant CLAIM_EXPIRY = 365 days;
    uint256 public constant FREE_TIER_WEIGHT = 2_000;          // 0.2x = 2000 / 10000

    // ── State ─────────────────────────────────────────────────────────
    uint256 public totalMinted;
    uint256 public immutable genesisTimestamp;

    // epoch => node => effective gpu-seconds
    mapping(uint256 => mapping(address => uint256)) public epochWork;
    mapping(uint256 => uint256) public epochTotalWork;
    mapping(uint256 => uint256) public epochNodeCount;
    mapping(uint256 => mapping(address => bool)) public claimed;
    mapping(uint256 => mapping(address => bytes32)) public workProofRoots;

    // ── Events ────────────────────────────────────────────────────────
    event WorkSubmitted(uint256 indexed epoch, address indexed node, uint256 effectiveWork, bytes32 proofRoot);
    event Claimed(uint256 indexed epoch, address indexed node, uint256 amount);
    event WorkChallenged(uint256 indexed epoch, address indexed miner, address indexed challenger);

    // ── Constructor ───────────────────────────────────────────────────
    constructor() ERC20("Mesh", "MESH") {
        genesisTimestamp = block.timestamp;
    }

    function decimals() public pure override returns (uint8) {
        return 8;
    }

    // ── View helpers ──────────────────────────────────────────────────
    function currentEpoch() public view returns (uint256) {
        return (block.timestamp - genesisTimestamp) / EPOCH_DURATION;
    }

    function epochEnd(uint256 epoch) public view returns (uint256) {
        return genesisTimestamp + (epoch + 1) * EPOCH_DURATION;
    }

    function rewardForEpoch(uint256 epoch) public view returns (uint256) {
        uint256 era = epoch / HALVING_INTERVAL;
        if (era >= 64) return 0;
        return INITIAL_REWARD >> era;
    }

    /// @notice Maximum share any single node can claim from an epoch.
    ///         max(10%, 100%/nodeCount) — doesn't punish small networks.
    function maxShareBps(uint256 epoch) public view returns (uint256) {
        uint256 n = epochNodeCount[epoch];
        if (n == 0) return 10_000;
        uint256 fair = 10_000 / n;
        return fair > 1_000 ? fair : 1_000;
    }

    // ── Mining: submit work ───────────────────────────────────────────

    /// @notice Submit GPU-seconds for a past epoch.
    /// @param epoch           The closed epoch to submit work for
    /// @param paidGpuSeconds  GPU-seconds backed by payment channel signatures (1.0x)
    /// @param freeGpuSeconds  GPU-seconds from free-tier requests (0.2x)
    /// @param consumerProofRoot Merkle root of (consumer, gpuSeconds, paid) leaves
    function submitWork(
        uint256 epoch,
        uint256 paidGpuSeconds,
        uint256 freeGpuSeconds,
        bytes32 consumerProofRoot
    ) external {
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

    // ── Mining: claim ─────────────────────────────────────────────────

    /// @notice Claim mined MESH for a past epoch (after submission window closes).
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

        // Dynamic cap
        uint256 cap = (reward * maxShareBps(epoch)) / 10_000;
        if (share > cap) share = cap;

        totalMinted += share;
        _mint(msg.sender, share);

        emit Claimed(epoch, msg.sender, share);
    }

    /// @notice Batch claim across multiple epochs in one tx.
    function claimBatch(uint256[] calldata epochs) external {
        for (uint256 i = 0; i < epochs.length; i++) {
            claim(epochs[i]);
        }
    }

    // ── Challenge ─────────────────────────────────────────────────────

    /// @notice Challenge fraudulent work within submission window.
    ///         Provide a Merkle proof showing a leaf in the miner's tree is invalid.
    /// @param epoch       The epoch of the work being challenged
    /// @param miner       The miner whose work is being challenged
    /// @param proof       Merkle proof for the fraudulent leaf
    /// @param leaf        The leaf data (keccak256 of consumer work record)
    function challengeWork(
        uint256 epoch,
        address miner,
        bytes32[] calldata proof,
        bytes32 leaf
    ) external {
        require(block.timestamp <= epochEnd(epoch) + SUBMISSION_WINDOW, "challenge window closed");
        require(workProofRoots[epoch][miner] != bytes32(0), "no submission");

        // Verify the leaf is in the miner's Merkle tree
        require(MerkleProof.verify(proof, workProofRoots[epoch][miner], leaf), "invalid proof");

        // If we reach here, the challenger has proven a leaf exists.
        // Off-chain verification confirms the leaf is fraudulent.
        // Slash the miner's work for this epoch.
        uint256 slashedWork = epochWork[epoch][miner];
        epochTotalWork[epoch] -= slashedWork;
        epochWork[epoch][miner] = 0;
        epochNodeCount[epoch] -= 1;

        emit WorkChallenged(epoch, miner, msg.sender);
    }
}
