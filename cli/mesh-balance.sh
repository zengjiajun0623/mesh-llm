#!/bin/bash
# mesh-balance.sh — Check MESH token balance and claimable rewards
# Usage: ./mesh-balance.sh [address]

set -euo pipefail

MESH_TOKEN="${MESH_TOKEN:-0x5f74F34113AE4C47A4e3e8Bdde7BC02121B4480c}"
RPC="${RPC_URL:-https://ethereum-sepolia-rpc.publicnode.com}"
ADDR="${1:-0x555AB612C66643cb0Edf06C2990C3Baeb33C26eF}"  # default: mesh-dev

echo "=== MESH Balance ==="
echo "Token:   $MESH_TOKEN"
echo "Address: $ADDR"
echo ""

# Current balance
BAL_RAW=$(cast call "$MESH_TOKEN" "balanceOf(address)(uint256)" "$ADDR" --rpc-url "$RPC" | awk '{print $1}')
BAL=$(python3 -c "print(f'{int(\"$BAL_RAW\") / 1e8:.8f} MESH')")
echo "Balance: $BAL"

# Current epoch
CURRENT=$(cast call "$MESH_TOKEN" "currentEpoch()(uint256)" --rpc-url "$RPC")
echo "Current epoch: $CURRENT"

# Total minted
MINTED_RAW=$(cast call "$MESH_TOKEN" "totalMinted()(uint256)" --rpc-url "$RPC" | awk '{print $1}')
MINTED=$(python3 -c "print(f'{int(\"$MINTED_RAW\") / 1e8:.8f} MESH')")
echo "Total minted:  $MINTED"

# Check work submitted for recent epochs
echo ""
echo "=== Epoch Work ==="
for e in $(seq 0 $((CURRENT > 5 ? CURRENT - 5 : 0)) $((CURRENT - 1))); do
    WORK=$(cast call "$MESH_TOKEN" "epochWork(uint256,address)(uint256)" "$e" "$ADDR" --rpc-url "$RPC" 2>/dev/null || echo "0")
    TOTAL=$(cast call "$MESH_TOKEN" "epochTotalWork(uint256)(uint256)" "$e" --rpc-url "$RPC" 2>/dev/null || echo "0")
    CLAIMED=$(cast call "$MESH_TOKEN" "claimed(uint256,address)(bool)" "$e" "$ADDR" --rpc-url "$RPC" 2>/dev/null || echo "false")
    REWARD_RAW=$(cast call "$MESH_TOKEN" "rewardForEpoch(uint256)(uint256)" "$e" --rpc-url "$RPC" 2>/dev/null || echo "0")

    if [ "$WORK" != "0" ]; then
        SHARE=$(python3 -c "
w = int('$WORK'); t = int('$TOTAL'); r = int('$REWARD_RAW')
share = (r * w // t) if t > 0 else 0
print(f'{share / 1e8:.8f} MESH')
")
        echo "  Epoch $e: work=$WORK/$TOTAL claimed=$CLAIMED reward=$SHARE"
    fi
done
