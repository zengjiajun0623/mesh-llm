// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

/// @title PaymentChannel — Unidirectional micropayment channels for inference
/// @notice Consumer deposits MESH, signs incrementing balance updates off-chain,
///         provider settles on-chain when profitable.
contract PaymentChannel {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    struct Channel {
        address consumer;
        address provider;
        uint256 deposit;
        uint256 spent;         // last settled cumulative amount
        uint256 nonce;
        uint48 expiry;
        bool open;
    }

    IERC20 public immutable meshToken;
    uint256 public nextChannelId;
    mapping(uint256 => Channel) public channels;

    // Challenge period for disputes
    uint256 public constant CHALLENGE_PERIOD = 1 days;

    // Pending settlements (for dispute window)
    struct PendingSettlement {
        uint256 amount;
        uint256 nonce;
        uint256 timestamp;
    }
    mapping(uint256 => PendingSettlement) public pendingSettlements;

    event ChannelOpened(uint256 indexed channelId, address indexed consumer, address indexed provider, uint256 deposit, uint48 expiry);
    event ChannelSettled(uint256 indexed channelId, uint256 amount);
    event ChannelChallenged(uint256 indexed channelId, uint256 amount, uint256 nonce);
    event ChannelClosed(uint256 indexed channelId, uint256 refund);
    event ChannelTopUp(uint256 indexed channelId, uint256 addedDeposit);

    constructor(address _meshToken) {
        meshToken = IERC20(_meshToken);
    }

    /// @notice Open a payment channel with a provider.
    /// @param provider  The node operator's address
    /// @param deposit   Amount of MESH to lock
    /// @param duration  Channel lifetime in seconds
    function open(address provider, uint256 deposit, uint48 duration) external returns (uint256 channelId) {
        require(provider != address(0), "zero provider");
        require(provider != msg.sender, "self channel");
        require(deposit > 0, "zero deposit");
        require(duration > 0, "zero duration");

        meshToken.transferFrom(msg.sender, address(this), deposit);

        channelId = nextChannelId++;
        channels[channelId] = Channel({
            consumer: msg.sender,
            provider: provider,
            deposit: deposit,
            spent: 0,
            nonce: 0,
            expiry: uint48(block.timestamp) + duration,
            open: true
        });

        emit ChannelOpened(channelId, msg.sender, provider, deposit, uint48(block.timestamp) + duration);
    }

    /// @notice Top up an existing channel with more MESH.
    function topUp(uint256 channelId, uint256 amount) external {
        Channel storage ch = channels[channelId];
        require(ch.open, "channel closed");
        require(msg.sender == ch.consumer, "not consumer");
        require(amount > 0, "zero amount");

        meshToken.transferFrom(msg.sender, address(this), amount);
        ch.deposit += amount;

        emit ChannelTopUp(channelId, amount);
    }

    /// @notice Provider settles with the latest signed state from consumer.
    /// @param channelId       The channel to settle
    /// @param cumulativeAmount Total amount consumer has agreed to pay (incrementing)
    /// @param nonce           State nonce (must be > last settled nonce)
    /// @param signature       Consumer's signature over (channelId, cumulativeAmount, nonce)
    function settle(
        uint256 channelId,
        uint256 cumulativeAmount,
        uint256 nonce,
        bytes calldata signature
    ) external {
        Channel storage ch = channels[channelId];
        require(ch.open, "channel closed");
        require(msg.sender == ch.provider, "not provider");
        require(cumulativeAmount > ch.spent, "nothing new to settle");
        require(cumulativeAmount <= ch.deposit, "exceeds deposit");
        require(nonce > ch.nonce, "stale nonce");

        // Verify consumer signed this state
        bytes32 stateHash = keccak256(abi.encode(channelId, cumulativeAmount, nonce));
        bytes32 ethHash = stateHash.toEthSignedMessageHash();
        address signer = ethHash.recover(signature);
        require(signer == ch.consumer, "invalid signature");

        uint256 payment = cumulativeAmount - ch.spent;
        ch.spent = cumulativeAmount;
        ch.nonce = nonce;

        meshToken.transfer(ch.provider, payment);

        emit ChannelSettled(channelId, payment);
    }

    /// @notice Consumer reclaims remaining deposit after expiry.
    function expire(uint256 channelId) external {
        Channel storage ch = channels[channelId];
        require(ch.open, "channel closed");
        require(block.timestamp >= ch.expiry, "not expired");
        require(msg.sender == ch.consumer, "not consumer");

        ch.open = false;
        uint256 refund = ch.deposit - ch.spent;
        if (refund > 0) {
            meshToken.transfer(ch.consumer, refund);
        }

        emit ChannelClosed(channelId, refund);
    }

    /// @notice Provider can close the channel after settling all owed amounts.
    function closeByProvider(
        uint256 channelId,
        uint256 cumulativeAmount,
        uint256 nonce,
        bytes calldata signature
    ) external {
        Channel storage ch = channels[channelId];
        require(ch.open, "channel closed");
        require(msg.sender == ch.provider, "not provider");

        // Settle final state if new
        if (cumulativeAmount > ch.spent && nonce > ch.nonce) {
            bytes32 stateHash = keccak256(abi.encode(channelId, cumulativeAmount, nonce));
            bytes32 ethHash = stateHash.toEthSignedMessageHash();
            address signer = ethHash.recover(signature);
            require(signer == ch.consumer, "invalid signature");

            uint256 payment = cumulativeAmount - ch.spent;
            ch.spent = cumulativeAmount;
            ch.nonce = nonce;
            meshToken.transfer(ch.provider, payment);
        }

        // Close and refund remaining to consumer
        ch.open = false;
        uint256 refund = ch.deposit - ch.spent;
        if (refund > 0) {
            meshToken.transfer(ch.consumer, refund);
        }

        emit ChannelClosed(channelId, refund);
    }

    // ── View helpers ──────────────────────────────────────────────────

    function channelBalance(uint256 channelId) external view returns (uint256) {
        Channel storage ch = channels[channelId];
        return ch.deposit - ch.spent;
    }

    function isOpen(uint256 channelId) external view returns (bool) {
        return channels[channelId].open;
    }
}
