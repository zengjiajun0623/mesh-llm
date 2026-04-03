// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/// @title MeshPaymaster — Sponsors gas for MESH holders (ERC-4337 compatible)
/// @notice Allows miners to claim tokens and interact with payment channels
///         without holding ETH. Deducts a MESH fee for gas sponsorship.
/// @dev This is a simplified paymaster for testnet. Production version
///      should inherit from a full ERC-4337 BasePaymaster implementation.
contract MeshPaymaster is Ownable {
    IERC20 public immutable meshToken;

    /// @notice MESH per unit of gas (scaled by 1e18). Set by oracle or owner.
    uint256 public meshPerGas;

    /// @notice Fee markup in basis points (10000 = 1x, 11000 = 1.1x).
    uint256 public feeMarkup = 11_000; // 10% markup

    /// @notice Minimum MESH balance to be eligible for gas sponsorship.
    uint256 public minMeshBalance;

    event GasSponsored(address indexed user, uint256 gasCost, uint256 meshFee);
    event MeshPerGasUpdated(uint256 newRate);
    event FeeMarkupUpdated(uint256 newMarkup);
    event Withdrawn(address indexed to, uint256 ethAmount, uint256 meshAmount);

    constructor(address _meshToken, uint256 _meshPerGas) Ownable(msg.sender) {
        meshToken = IERC20(_meshToken);
        meshPerGas = _meshPerGas;
    }

    /// @notice Calculate MESH cost for a given gas amount.
    function estimateMeshCost(uint256 gasAmount, uint256 gasPrice) public view returns (uint256) {
        uint256 ethCost = gasAmount * gasPrice;
        return (ethCost * meshPerGas * feeMarkup) / (1e18 * 10_000);
    }

    /// @notice Update the MESH/gas exchange rate. Called by owner or oracle.
    function setMeshPerGas(uint256 _meshPerGas) external onlyOwner {
        meshPerGas = _meshPerGas;
        emit MeshPerGasUpdated(_meshPerGas);
    }

    /// @notice Update fee markup.
    function setFeeMarkup(uint256 _feeMarkup) external onlyOwner {
        require(_feeMarkup >= 10_000, "markup below 1x");
        require(_feeMarkup <= 20_000, "markup above 2x");
        feeMarkup = _feeMarkup;
        emit FeeMarkupUpdated(_feeMarkup);
    }

    /// @notice Update minimum MESH balance for sponsorship eligibility.
    function setMinMeshBalance(uint256 _min) external onlyOwner {
        minMeshBalance = _min;
    }

    /// @notice Fund the paymaster with ETH for gas sponsorship.
    receive() external payable {}

    /// @notice Withdraw ETH and/or MESH collected as fees.
    function withdraw(address to, uint256 ethAmount, uint256 meshAmount) external onlyOwner {
        if (ethAmount > 0) {
            (bool ok,) = to.call{value: ethAmount}("");
            require(ok, "ETH transfer failed");
        }
        if (meshAmount > 0) {
            meshToken.transfer(to, meshAmount);
        }
        emit Withdrawn(to, ethAmount, meshAmount);
    }

    // ────────────────────────────────────────────────────────────────
    // ERC-4337 integration points (simplified for testnet)
    //
    // In production, this contract inherits from the canonical
    // BasePaymaster and implements:
    //   - _validatePaymasterUserOp: check user has enough MESH, lock it
    //   - _postOp: calculate actual gas cost, deduct actual MESH fee
    //
    // The simplified version below provides the accounting logic
    // that a full implementation would call.
    // ────────────────────────────────────────────────────────────────

    /// @notice Pre-validate: check if user has enough MESH to cover estimated gas.
    /// @dev Called by the ERC-4337 entrypoint during validation.
    function validateSponsor(address user, uint256 maxGas, uint256 gasPrice) external view returns (bool) {
        if (address(this).balance == 0) return false;
        uint256 meshCost = estimateMeshCost(maxGas, gasPrice);
        uint256 balance = meshToken.balanceOf(user);
        return balance >= meshCost && balance >= minMeshBalance;
    }

    /// @notice Post-op: deduct actual MESH fee from user after tx executes.
    /// @dev Called by the ERC-4337 entrypoint after UserOp execution.
    function chargeMeshFee(address user, uint256 actualGas, uint256 gasPrice) external returns (uint256 meshFee) {
        meshFee = estimateMeshCost(actualGas, gasPrice);
        meshToken.transferFrom(user, address(this), meshFee);
        emit GasSponsored(user, actualGas * gasPrice, meshFee);
    }
}
