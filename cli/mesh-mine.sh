#!/bin/bash
# mesh-mine.sh — Submit mining work for a completed epoch via Elytro
# Usage: ./mesh-mine.sh <epoch> <paid_gpu_seconds> <free_gpu_seconds>
#
# Example: ./mesh-mine.sh 0 0 5000   (submit 5000 free-tier GPU-seconds for epoch 0)

set -euo pipefail

MESH_TOKEN="${MESH_TOKEN:-0x5f74F34113AE4C47A4e3e8Bdde7BC02121B4480c}"
ELYTRO_ACCOUNT="${ELYTRO_ACCOUNT:-mesh-dev}"
RPC="${RPC_URL:-https://ethereum-sepolia-rpc.publicnode.com}"

EPOCH="${1:?Usage: mesh-mine.sh <epoch> <paid_gpu_seconds> <free_gpu_seconds>}"
PAID="${2:?Missing paid_gpu_seconds}"
FREE="${3:?Missing free_gpu_seconds}"

echo "=== MESH Mining ==="
echo "Token:   $MESH_TOKEN"
echo "Account: $ELYTRO_ACCOUNT"
echo "Epoch:   $EPOCH"
echo "Paid GPU-seconds:  $PAID (1.0x weight)"
echo "Free GPU-seconds:  $FREE (0.2x weight)"

# Check current epoch
CURRENT=$(cast call "$MESH_TOKEN" "currentEpoch()(uint256)" --rpc-url "$RPC")
echo "Current epoch:     $CURRENT"

if [ "$EPOCH" -ge "$CURRENT" ]; then
    echo "ERROR: Epoch $EPOCH is not closed yet (current: $CURRENT). Wait for it to end."
    exit 1
fi

# Build proof root (simplified: hash of epoch + sender + work)
PROOF_ROOT=$(cast keccak "epoch${EPOCH}_paid${PAID}_free${FREE}")
echo "Proof root:        $PROOF_ROOT"

# Encode the submitWork call
CALLDATA=$(cast calldata "submitWork(uint256,uint256,uint256,bytes32)" "$EPOCH" "$PAID" "$FREE" "$PROOF_ROOT")

echo ""
echo "Submitting work via Elytro..."
elytro tx send "$ELYTRO_ACCOUNT" --tx "to:${MESH_TOKEN},data:${CALLDATA}"
echo ""
echo "Work submitted for epoch $EPOCH!"
