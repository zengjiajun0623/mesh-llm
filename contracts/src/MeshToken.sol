// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "solmate/tokens/ERC20.sol";

/// @title MeshToken — Earn by serving free AI inference
/// @notice Nodes serve free inference, consumers sign receipts,
///         nodes submit receipts on-chain to mint MESH.
contract MeshToken is ERC20 {
    uint256 public constant MAX_SUPPLY = 21_000_000e18;
    uint256 public constant REWARD_PER_RECEIPT = 1e18; // 1 MESH per valid receipt
    uint256 public constant RATE_LIMIT = 1 hours;
    uint256 public constant RECEIPT_EXPIRY = 7 days;
    uint256 public constant MAX_BATCH = 100;

    uint256 public totalMinted;

    mapping(bytes32 => bool) public usedReceipts;
    mapping(address => uint256) public lastReceiptTime;

    event Claimed(address indexed node, address indexed consumer, bytes32 receiptHash);

    constructor() ERC20("Mesh", "MESH", 18) {}

    /// @notice Claim MESH by submitting consumer-signed receipts.
    function claim(
        address[] calldata consumers,
        uint256[] calldata timestamps,
        bytes[] calldata signatures
    ) external {
        require(consumers.length == timestamps.length && consumers.length == signatures.length, "length mismatch");
        require(consumers.length > 0 && consumers.length <= MAX_BATCH, "bad batch size");

        uint256 validCount = 0;

        for (uint256 i = 0; i < consumers.length; i++) {
            bytes32 receiptHash = keccak256(abi.encode(msg.sender, consumers[i], timestamps[i]));

            if (usedReceipts[receiptHash]) continue;
            if (consumers[i] == msg.sender) continue;
            if (timestamps[i] > block.timestamp) continue;
            if (block.timestamp - timestamps[i] > RECEIPT_EXPIRY) continue;
            if (lastReceiptTime[consumers[i]] != 0 && timestamps[i] < lastReceiptTime[consumers[i]] + RATE_LIMIT) continue;

            // Verify signature: consumer signed (node, consumer, timestamp)
            bytes32 ethHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", receiptHash));
            address signer = _recover(ethHash, signatures[i]);
            if (signer != consumers[i]) continue;

            usedReceipts[receiptHash] = true;
            lastReceiptTime[consumers[i]] = timestamps[i];
            validCount++;

            emit Claimed(msg.sender, consumers[i], receiptHash);
        }

        require(validCount > 0, "no valid receipts");

        uint256 reward = validCount * REWARD_PER_RECEIPT;
        uint256 remaining = MAX_SUPPLY - totalMinted;
        if (reward > remaining) reward = remaining;
        require(reward > 0, "max supply reached");

        totalMinted += reward;
        _mint(msg.sender, reward);
    }

    function mintableSupply() external view returns (uint256) {
        return MAX_SUPPLY - totalMinted;
    }

    function _recover(bytes32 hash, bytes calldata sig) internal pure returns (address) {
        require(sig.length == 65, "bad sig length");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := calldataload(sig.offset)
            s := calldataload(add(sig.offset, 32))
            v := byte(0, calldataload(add(sig.offset, 64)))
        }
        return ecrecover(hash, v, r, s);
    }
}
