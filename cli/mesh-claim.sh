#!/bin/bash
# mesh-claim.sh — Claim mined MESH tokens for one or more epochs via Elytro
# Usage: ./mesh-claim.sh <epoch>          (single epoch)
#        ./mesh-claim.sh <e1> <e2> <e3>   (batch claim multiple epochs)

set -euo pipefail

MESH_TOKEN="${MESH_TOKEN:-0x5f74F34113AE4C47A4e3e8Bdde7BC02121B4480c}"
ELYTRO_ACCOUNT="${ELYTRO_ACCOUNT:-mesh-dev}"
RPC="${RPC_URL:-https://ethereum-sepolia-rpc.publicnode.com}"

if [ $# -eq 0 ]; then
    echo "Usage: mesh-claim.sh <epoch> [epoch2] [epoch3] ..."
    exit 1
fi

ADDR=$(elytro account list "$ELYTRO_ACCOUNT" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['accounts'][0]['address'])" 2>/dev/null || echo "0x555AB612C66643cb0Edf06C2990C3Baeb33C26eF")

echo "=== MESH Claim ==="
echo "Token:   $MESH_TOKEN"
echo "Account: $ELYTRO_ACCOUNT ($ADDR)"

# Check balance before
BAL_BEFORE=$(cast call "$MESH_TOKEN" "balanceOf(address)(uint256)" "$ADDR" --rpc-url "$RPC")

if [ $# -eq 1 ]; then
    EPOCH="$1"
    echo "Claiming epoch: $EPOCH"

    CALLDATA=$(cast calldata "claim(uint256)" "$EPOCH")
    elytro tx send "$ELYTRO_ACCOUNT" --tx "to:${MESH_TOKEN},data:${CALLDATA}"
else
    # Batch claim
    EPOCHS_ARG="["
    for e in "$@"; do
        EPOCHS_ARG="${EPOCHS_ARG}${e},"
    done
    EPOCHS_ARG="${EPOCHS_ARG%,}]"
    echo "Batch claiming epochs: $@"

    CALLDATA=$(cast calldata "claimBatch(uint256[])" "$EPOCHS_ARG")
    elytro tx send "$ELYTRO_ACCOUNT" --tx "to:${MESH_TOKEN},data:${CALLDATA}"
fi

echo ""
# Check balance after
BAL_AFTER=$(cast call "$MESH_TOKEN" "balanceOf(address)(uint256)" "$ADDR" --rpc-url "$RPC")
EARNED=$(python3 -c "print(f'{(int(\"$BAL_AFTER\") - int(\"$BAL_BEFORE\")) / 1e8:.8f} MESH')")
echo "Earned: $EARNED"
echo "New balance: $(python3 -c "print(f'{int(\"$BAL_AFTER\") / 1e8:.8f} MESH')")"
