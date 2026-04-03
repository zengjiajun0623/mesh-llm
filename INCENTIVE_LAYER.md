# Incentive Layer Architecture

Ethereum-based incentive layer for mesh-llm, enabling node operators to earn tokens for serving inference and consumers to pay for usage. Uses Elytro wallet (ERC-4337) as the default wallet.

## Design Principles

1. **Plugin, not fork** — Build as a mesh-llm plugin using the existing plugin protocol. No changes to core mesh-llm code. Stays upstream-compatible.
2. **Off-chain first, settle on-chain** — Inference is fast (milliseconds). Ethereum is slow (seconds). Use payment channels for per-request accounting, batch-settle on-chain.
3. **Elytro-native** — ERC-4337 account abstraction means agents can pay for inference without holding EOA keys. On-chain policy enforcement (spending limits, allowlists) gives users control over agent spending.
4. **Opt-in** — Nodes without the plugin operate exactly as today. The incentive layer is additive.
5. **Zero ETH to start** — ERC-4337 paymaster sponsors wallet deployment and first claim. Miners never need to acquire ETH. The paymaster deducts a small MESH fee from minted tokens.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Consumer Agent                      │
│            (Goose, Claude Code, custom app)              │
│                          │                               │
│                   Elytro Wallet (ERC-4337)               │
│            (spending limits, session keys)                │
└──────────────────────────┬──────────────────────────────┘
                           │ OpenAI-compatible API
                           │ + X-Mesh-Payment header
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     mesh-llm node                        │
│  ┌───────────────┐  ┌─────────────────────────────────┐ │
│  │  API Proxy     │──│  Incentive Plugin               │ │
│  │  :9337         │  │                                 │ │
│  └───────────────┘  │  ┌───────────┐ ┌─────────────┐ │ │
│                      │  │ Metering  │ │ Payment     │ │ │
│  ┌───────────────┐  │  │ Engine    │ │ Channel Mgr │ │ │
│  │  llama-server  │  │  └───────────┘ └──────┬──────┘ │ │
│  │  (inference)   │  │                       │        │ │
│  └───────────────┘  │  ┌─────────────────────┴──────┐ │ │
│                      │  │ Settlement (batch on-chain) │ │ │
│                      │  └────────────────────────────┘ │ │
│                      └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼ batch settle
┌─────────────────────────────────────────────────────────┐
│                   Ethereum Mainnet                       │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ MESH Token   │  │ Payment      │  │ Paymaster     │ │
│  │ (ERC-20)     │  │ Channel      │  │ (gas sponsor) │ │
│  │              │  │              │  │               │ │
│  └──────────────┘  └──────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. MESH Token (ERC-20) — Bitcoin-style Issuance

ERC-20 on Ethereum mainnet with a **fixed supply and halving schedule**, mined by contributing GPU compute to the mesh network.

**Supply Model (Bitcoin-inspired):**

| Parameter | Value |
|---|---|
| Max supply | 21,000,000 MESH |
| Initial epoch reward | 7,192 MESH per day |
| Epoch length | 1 day (24 hours, 00:00 UTC) |
| Halving interval | 1,460 epochs (~4 years) |
| Minimum reward | 0.00000001 MESH |
| Decimal places | 8 (like BTC) |

**Halving schedule:**

| Era | Years | Reward/Day | Era Total | Cumulative |
|---|---|---|---|---|
| 1 | 0–4 | 7,192 MESH | 10,500,320 | 10,500,320 |
| 2 | 4–8 | 3,596 MESH | 5,250,160 | 15,750,480 |
| 3 | 8–12 | 1,798 MESH | 2,625,080 | 18,375,560 |
| 4 | 12–16 | 899 MESH | 1,312,540 | 19,688,100 |
| 5 | 16–20 | ~450 MESH | ~656,000 | ~20,344,100 |
| ... | ... | ... | ... | → 21,000,000 |

~50% mined in the first 4 years, ~75% in 8 years, asymptotically approaches 21M. Same curve as Bitcoin.

> **Note on effective supply:** If nodes fail to claim (go offline, lose keys), those tokens are never minted. Actual circulating supply may be less than 21M — same property as Bitcoin with lost coins. 21M is a ceiling, not a guaranteed target.

**Mining = Serving Inference (Proof-of-Useful-Work):**

Instead of hashing, nodes "mine" MESH by serving inference requests on the mesh network. The work product is actual AI inference, not wasted energy.

**Single mining mechanism: GPU-seconds.**

Nodes earn MESH proportional to the GPU time they spend serving real inference requests. No model weight tables, no oracles, no governance over which models are "worth more."

```
node_reward = (node_gpu_seconds / total_network_gpu_seconds) × daily_epoch_reward
```

That's it. Serve inference, burn GPU time, earn tokens.

**How GPU-seconds are measured:**

The plugin tracks wall-clock time from request start to response end on llama-server. This is local, objective, and model-agnostic.

- A 70B model taking 8 seconds to generate 100 tokens = 8 GPU-seconds
- A 3B model generating 100 tokens in 0.3 seconds = 0.3 GPU-seconds
- A brand new model that drops tomorrow = just works, no config needed

Bigger models naturally earn more because they take longer per token. No one decides what a model is "worth" — the GPU clock decides.

**Two tiers of work proof:**

Work proofs differ based on whether the consumer paid or not:

- **Paid requests (full weight):** The payment channel signature IS the work proof. A consumer who signed an incrementing payment state for a specific provider proves they requested and valued the work. Cannot be faked without real token transfer. These count at **1.0x** weight toward mining.

- **Free-tier requests (reduced weight):** Requests without payment have no cryptographic proof of genuine demand. These count at **0.2x** weight toward mining. This still incentivizes free serving (bootstraps network) but makes self-dealing 5x less profitable than actually attracting paying consumers.

```
effective_gpu_seconds = paid_gpu_seconds × 1.0 + free_gpu_seconds × 0.2
```

**How a daily epoch works:**

```
1. Epoch opens at 00:00 UTC

2. During the day, nodes serve inference
   - Plugin tracks locally: GPU-seconds per request, paid vs free, consumer address
   - Work summaries gossiped between peers via ChannelMessage for transparency

3. Epoch closes at 00:00 UTC next day

4. Submission window: 3 days after epoch close
   - Nodes submit work claims with Merkle root of consumer proofs
   - Multiple nodes submitting doesn't race — each submits independently

5. Claim window opens: day 3 after epoch close
   - All submissions are in, epochTotalWork is final
   - Nodes claim their proportional share
   - Claim window: 365 days (unclaimed after that = never minted)

6. Tokens minted to node operator's Elytro wallet via paymaster
```

```solidity
contract MeshToken is ERC20 {
    uint256 public constant MAX_SUPPLY = 21_000_000e8;       // 8 decimals
    uint256 public constant INITIAL_REWARD = 7_192e8;         // 7,192 MESH per epoch
    uint256 public constant HALVING_INTERVAL = 1_460;         // ~4 years of daily epochs
    uint256 public constant EPOCH_DURATION = 1 days;
    uint256 public constant SUBMISSION_WINDOW = 3 days;       // submit work within 3 days
    uint256 public constant CLAIM_EXPIRY = 365 days;          // must claim within 1 year

    uint256 public totalMinted;
    uint256 public genesisTimestamp;

    // epoch => node => effective gpu-seconds submitted
    mapping(uint256 => mapping(address => uint256)) public epochWork;
    mapping(uint256 => uint256) public epochTotalWork;
    mapping(uint256 => mapping(address => bool)) public claimed;

    // epoch => node => merkle root of consumer proofs
    mapping(uint256 => mapping(address => bytes32)) public workProofRoots;

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

    /// Submit work for a past epoch.
    /// paidGpuSeconds: GPU-seconds backed by payment channel signatures (1.0x weight)
    /// freeGpuSeconds: GPU-seconds from free-tier requests (0.2x weight)
    /// consumerProofRoot: Merkle root of (consumer, gpuSeconds, paymentSig) tuples
    function submitWork(
        uint256 epoch,
        uint256 paidGpuSeconds,
        uint256 freeGpuSeconds,
        bytes32 consumerProofRoot
    ) external {
        require(epoch < currentEpoch(), "epoch not closed");
        require(block.timestamp <= epochEnd(epoch) + SUBMISSION_WINDOW, "submission window closed");
        require(workProofRoots[epoch][msg.sender] == bytes32(0), "already submitted");

        // Effective GPU-seconds: paid at full weight, free at 0.2x
        uint256 effectiveWork = paidGpuSeconds + (freeGpuSeconds / 5);

        workProofRoots[epoch][msg.sender] = consumerProofRoot;
        epochWork[epoch][msg.sender] = effectiveWork;
        epochTotalWork[epoch] += effectiveWork;
    }

    /// Claim mined MESH for a past epoch. Only after submission window closes.
    function claim(uint256 epoch) external {
        require(block.timestamp > epochEnd(epoch) + SUBMISSION_WINDOW, "submission window open");
        require(block.timestamp <= epochEnd(epoch) + CLAIM_EXPIRY, "claim expired");
        require(!claimed[epoch][msg.sender], "already claimed");
        require(epochWork[epoch][msg.sender] > 0, "no work");

        claimed[epoch][msg.sender] = true;

        uint256 reward = rewardForEpoch(epoch);
        if (totalMinted + reward > MAX_SUPPLY) {
            reward = MAX_SUPPLY - totalMinted;
        }
        if (reward == 0) return;

        // Node's share — dynamic cap based on node count
        uint256 nodeWork = epochWork[epoch][msg.sender];
        uint256 totalWork = epochTotalWork[epoch];
        uint256 share = (reward * nodeWork) / totalWork;

        // Cap: max(10%, 100%/nodeCount) — prevents whale dominance
        // but doesn't punish small networks
        uint256 nodeCount = epochNodeCount[epoch];
        uint256 maxBps = nodeCount <= 10 ? 10000 / nodeCount : 1000;
        uint256 maxShare = (reward * maxBps) / 10000;
        if (share > maxShare) share = maxShare;

        _mint(msg.sender, share);
        totalMinted += share;
    }

    /// Batch claim multiple epochs in one tx.
    function claimBatch(uint256[] calldata epochs) external {
        for (uint256 i = 0; i < epochs.length; i++) {
            this.claim(epochs[i]);
        }
    }

    /// Challenge a fraudulent work submission (within submission window).
    /// Provide a Merkle proof that a specific consumer entry is invalid.
    function challengeWork(
        uint256 epoch,
        address miner,
        bytes32[] calldata merkleProof,
        address consumer,
        uint256 consumerGpuSeconds,
        bytes calldata paymentSig
    ) external {
        require(block.timestamp <= epochEnd(epoch) + SUBMISSION_WINDOW, "challenge window closed");

        // Verify the leaf is in the miner's Merkle root
        bytes32 leaf = keccak256(abi.encode(consumer, consumerGpuSeconds, paymentSig));
        require(verifyMerkleProof(merkleProof, workProofRoots[epoch][miner], leaf), "not in tree");

        // Verify the payment signature is invalid (for paid claims)
        // or consumer doesn't exist / didn't actually request
        // If challenge succeeds: slash miner's work for this epoch
        epochTotalWork[epoch] -= epochWork[epoch][miner];
        epochWork[epoch][miner] = 0;

        // Reward challenger with a portion of the slashed work's value
        // (challenger gets to submit equivalent work credit)
    }
}
```

### 2. Payment Channel Contract

Unidirectional payment channels for per-request micropayments. Consumer opens a channel, signs incrementing balance updates off-chain, provider settles on-chain when ready.

Payment channels are the **spending mechanism** — how consumers pay for inference. The payment channel signature also serves as the **work proof** for full-weight mining. Clean dual purpose: pay for inference AND prove you did real work.

```solidity
struct Channel {
    address consumer;       // Elytro account paying for inference
    address provider;       // node operator receiving payment
    uint256 deposit;        // MESH tokens locked
    uint256 spent;          // last settled amount
    uint256 nonce;          // replay protection
    uint48 expiry;          // auto-refund deadline
}

function open(address provider, uint48 duration) external payable;
function settle(bytes calldata signedState) external;  // provider claims
function challenge(bytes calldata signedState) external; // dispute
function expire() external;                             // consumer reclaims after expiry
```

**Why payment channels over per-tx payments:**
- One inference request = ~0.001 MESH. On-chain tx on Ethereum mainnet = $2-10.
- Payment channels: open once, sign hundreds of state updates off-chain, settle once.
- Latency: signing an off-chain state update takes <1ms. On-chain tx takes seconds.
- Settlement frequency is economic: settle when accumulated value >> gas cost.

### 3. MeshPaymaster (ERC-4337 Gas Sponsor)

**The key to "zero ETH to start."**

New miners have MESH tokens but no ETH. The paymaster sponsors gas for their transactions and deducts a small MESH fee.

```solidity
contract MeshPaymaster is BasePaymaster {
    MeshToken public meshToken;
    uint256 public meshPerGas;          // MESH/gas exchange rate (oracle-fed)
    uint256 public feeMarkup = 1100;    // 10% markup in basis points

    function _validatePaymasterUserOp(
        PackedUserOperation calldata userOp,
        bytes32 userOpHash,
        uint256 maxCost
    ) internal override returns (bytes memory context, uint256 validationData) {
        uint256 meshCost = (maxCost * meshPerGas * feeMarkup) / 10000;
        require(meshToken.balanceOf(userOp.sender) >= meshCost);
        return (abi.encode(userOp.sender, meshCost), 0);
    }

    function _postOp(
        PostOpMode mode,
        bytes calldata context,
        uint256 actualGasCost,
        uint256 actualUserOpFeePerGas
    ) internal override {
        (address sender, uint256 maxMeshCost) = abi.decode(context, (address, uint256));
        uint256 actualMeshCost = (actualGasCost * meshPerGas * feeMarkup) / 10000;
        meshToken.transferFrom(sender, address(this), actualMeshCost);
    }
}
```

**Paymaster-sponsored operations:**
- First-ever claim (includes wallet deployment via CREATE2)
- Payment channel open/settle
- Any MESH transfer

**How it works for a new miner:**

```
Day 1: Alice runs mesh-llm --auto
  → Elytro address computed (counterfactual, not deployed)
  → Starts serving inference, accumulating GPU-seconds

Day 7: Alice claims her first week of mining
  → Plugin builds UserOp: deploy wallet + claim MESH from 4 epochs (3-day submission window)
  → Paymaster sponsors gas (~$3 in ETH)
  → Paymaster deducts ~0.02 MESH as fee (10% markup on gas cost)
  → Alice receives her mined MESH minus fee, wallet is deployed
  → Alice never touched ETH
```

### 4. Incentive Plugin (mesh-llm plugin)

A mesh-llm plugin (Rust binary) that hooks into the existing plugin protocol.

```
~/.mesh-llm/plugin/incentive
```

**Plugin responsibilities:**

| Function | How |
|---|---|
| **Auto-wallet** | Generate an Elytro wallet on first `mesh-llm --auto` join. Zero setup. |
| **Metering** | Track GPU-seconds per request, paid vs free, per consumer |
| **Mining** | Accumulate work per epoch, build Merkle tree, batch submit + claim |
| **Pricing** | Configurable rate per model per 1K tokens (operator sets their price) |
| **Payment verification** | Validate signed payment channel states on incoming requests |
| **Channel management** | Track open channels, balances, nonces |
| **Settlement** | Batch-submit channel states on-chain via Elytro wallet + paymaster |
| **Gossip** | Broadcast epoch work summaries to peers via ChannelMessage for transparency |

**Plugin protocol integration:**

- `MeshEvent::PEER_UP` → discover new peers, exchange wallet addresses
- `ChannelMessage` → exchange epoch work summaries and anomaly alerts
- `tools/call` → expose `balance`, `mine_status`, `open_channel`, `settle`, `send` as MCP tools

**Smart claiming logic:**

The plugin doesn't claim every epoch. It batches claims to minimize gas cost:

```
accumulated_reward = estimated share across all claimable epochs
estimated_gas_cost = current_gas_price × claim_gas_units × meshPerGas
claim_threshold = estimated_gas_cost × 3  # only claim when reward is 3x gas cost

if accumulated_reward > claim_threshold:
    submit_claim(all_unclaimed_epochs)  # batch into single UserOp via paymaster
```

Small miners might claim weekly or monthly, large miners claim daily. The paymaster makes this seamless — no ETH management needed regardless.

### 5. Automatic Wallet Generation

**Zero-friction onboarding — join the mesh, get a wallet, start mining.**

When a node operator runs `mesh-llm --auto` for the first time with the incentive plugin enabled, the plugin automatically:

1. **Generates a local keypair** — stored encrypted at `~/.mesh-llm/wallet/keystore.json`
2. **Computes Elytro address** — deterministic via CREATE2 (no on-chain tx needed yet)
3. **Starts mining immediately** — the node begins accumulating GPU-seconds from its first served request
4. **First claim deploys the wallet** — paymaster sponsors gas, wallet materializes on-chain, MESH arrives

```
$ mesh-llm --auto

🔍 Discovering meshes via Nostr...
✅ Joining: unnamed (5 nodes, 4 models)
💰 First run detected — generating Elytro wallet...
   Address: 0x7a3b...f912
   Keystore: ~/.mesh-llm/wallet/keystore.json
   ⚠ Back up your keystore! This controls your mined MESH tokens.
📋 Assigned to serve DeepSeek-R1-Distill-Qwen-32B-Q4_K_M
⛏ Mining active — earning MESH for every request served

  Wallet:  0x7a3b...f912
  Balance: 0.00000000 MESH (mining...)
  Epoch:   #412 (daily) | Reward: 7,192 MESH | Your share: pending
  API:     http://localhost:9337
  Console: http://localhost:3131
```

**Wallet lifecycle:**

```
First run:
  mesh-llm --auto
  └─ incentive plugin starts
     └─ no keystore found at ~/.mesh-llm/wallet/
        └─ generate secp256k1 keypair
        └─ encrypt with password (or auto-generated, stored in OS keychain)
        └─ compute counterfactual Elytro address (CREATE2)
        └─ save to ~/.mesh-llm/wallet/keystore.json
        └─ register peer_id ↔ address mapping via ChannelMessage gossip
        └─ start mining (off-chain GPU-second accumulation)

First claim (when accumulated reward > gas threshold):
  └─ plugin builds UserOp: deploy wallet + batch claim epochs
  └─ paymaster sponsors ETH gas, deducts MESH fee
  └─ wallet deployed on-chain, MESH minted to address
  └─ node is now fully on-chain

Subsequent runs:
  mesh-llm --auto
  └─ incentive plugin starts
     └─ keystore found, load keypair
     └─ resume mining with existing address
```

**Key storage:**

```
~/.mesh-llm/wallet/
├── keystore.json          # Encrypted private key (EIP-2335 format)
├── address                # Plain text Elytro address (for display)
└── backup-phrase.txt.enc  # Encrypted social recovery phrase (optional)
```

**Recovery options (via Elytro):**
- **Social recovery** — set guardian addresses who can rotate keys if lost
- **Export** — `mesh-llm wallet export` dumps the keystore for import into any Elytro client
- **Import** — `mesh-llm wallet import <keystore>` to use an existing Elytro wallet

**CLI wallet commands:**

```bash
mesh-llm wallet                    # show address + balance + mining stats
mesh-llm wallet export             # export keystore
mesh-llm wallet import <file>      # import existing Elytro wallet
mesh-llm wallet send <to> <amount> # send MESH to another address
mesh-llm wallet history            # mining + payment history
mesh-llm wallet backup             # set up social recovery guardians
mesh-llm wallet claim              # force-claim all pending epochs now
```

### 6. Token Economy — The Circular Loop

MESH tokens flow in a closed loop between miners and consumers. **Every participant is both.**

```
  ┌─────────── Mine ────────────┐
  │  (earn by serving inference) │
  ▼                              │
┌──────┐   spend MESH    ┌──────┴─────┐
│ Node │ ──────────────→  │ Other Nodes │
│  A   │ ← inference ──  │ (providers) │
│      │                  └────────────┘
│      │   earn MESH           │
│      │ ◄─────────────────────┘
└──────┘   (payment channels)

Node A mines MESH by serving model X (GPU-seconds)
  → spends mined MESH to use model Y from Node B (payment channel)
  → Node B earns MESH from Node A's payment
  → Node B also mines from GPU-seconds serving Node A (at full 1.0x weight — paid request)
  → ...
```

**Example scenario:**

```
Alice runs a MacBook Pro with 32GB — serves Qwen3-4B (free_tier = true)
  → GPU serves 500 requests/day, ~200 GPU-seconds total (all free tier)
  → effective work = 200 × 0.2 = 40 effective GPU-seconds
  → if network total is 20,000 effective GPU-seconds:
    Alice earns (40/20,000) × 7,192 = 14.4 MESH/day
  → claims weekly via paymaster (no ETH needed)
  → spends some MESH to use DeepSeek-R1-32B on Bob's node

Bob runs a workstation with 4x A100 — serves DeepSeek-R1-32B (paid tier)
  → GPU serves 100 requests/day, ~800 GPU-seconds (mostly paid requests)
  → effective work = 750 × 1.0 + 50 × 0.2 = 760 effective GPU-seconds
  → earns (760/20,000) × 7,192 = 273.3 MESH from mining
  → also earns MESH from payment channels
  → claims daily

Carol runs a server cluster — serves MiniMax-M2.5 (paid tier, high demand)
  → GPU serves 50 requests/day, ~1,000 GPU-seconds (all paid)
  → effective work = 1,000 effective GPU-seconds
  → earns (1,000/20,000) × 7,192 = 359.6 MESH from mining
  → earns additional MESH from payment channels
```

**Key properties:**
- **No money needed to start** — run a node, mine tokens (free tier at 0.2x), use them to access other models
- **No ETH needed ever** — paymaster sponsors all gas, deducts MESH
- **Self-bootstrapping** — free tier ensures mining works from day 1
- **Paid work earns 5x more** — strong incentive to attract real paying consumers
- **Model-agnostic** — no weight tables, no governance. GPU clock is the only measure.
- **No gatekeepers** — anyone with a GPU can join, mine, spend, and leave

### 7. Elytro as Default Wallet

Elytro is the wallet layer for all on-chain interactions.

**Why Elytro specifically:**
- **ERC-4337** — smart account means the auto-generated wallet can have recovery, spending limits, session keys without users understanding private keys
- **Counterfactual deployment** — wallet address exists before first tx. Node starts mining immediately, wallet deploys on-chain only when first claiming tokens (paymaster sponsors gas)
- **Paymaster-compatible** — Elytro natively supports ERC-4337 paymasters, enabling zero-ETH operations
- **Agent-native** — session keys let automated processes (the incentive plugin) sign txs within policy bounds without exposing the master key
- **Self-custodial** — user owns the key. mesh-llm never phones home. Aligns with mesh-llm's decentralized ethos.

**For consumers (agents using other nodes' models):**
- Agent gets a **session key** from the user's Elytro account with scoped permissions:
  - Spending limit: max N MESH per day
  - Allowlist: only interact with PaymentChannel and MESH token contracts
  - Expiry: session key valid for X hours
- Agent opens payment channels and signs state updates using the session key
- User never exposes master key. On-chain policy enforces limits even if agent is compromised.

**For providers (node operators mining + earning):**
- Elytro account auto-generated on first mesh join
- Receives mined tokens + payment channel settlements
- All gas paid via paymaster (MESH-denominated)
- Social recovery protects earnings (no single key loss = funds lost)

**ERC-4337 flow for a request:**

```
Agent wants inference from another node
  → checks local MESH balance (mined tokens)
  → checks open channel with provider (off-chain)
  → if no channel: UserOp via Elytro to open channel + deposit MESH
    → paymaster sponsors ETH gas, deducts MESH fee
  → sends request with signed payment state in X-Mesh-Payment header
  → provider verifies signature, serves inference
  → provider batches states, settles via Elytro UserOp when profitable
```

### 8. Pricing Model

```toml
# ~/.mesh-llm/config.toml
[[plugin]]
name = "incentive"
command = "~/.mesh-llm/plugin/incentive"
enabled = true

[plugin.config]
# Pricing per 1K tokens (in MESH)
default_price = 0.001
[plugin.config.model_prices]
"DeepSeek-R1-Distill-Qwen-32B-Q4_K_M" = 0.002
"MiniMax-M2.5-Q4_K_M" = 0.005
"Llama-3.2-1B-Instruct-Q4_K_M" = 0.0005

# Free tier: serve without requiring payment (mines at 0.2x weight)
free_tier = true

# Claim threshold multiplier (claim when reward > gas_cost × this)
claim_threshold_multiplier = 3

# Elytro wallet config
wallet_address = "0x..."  # auto-generated if empty
chain_id = 1  # Ethereum mainnet
rpc_url = "https://eth.llamarpc.com"
```

## Request Flow

```
1. Agent sends chat completion request
   POST /v1/chat/completions
   X-Mesh-Payment: {channel_id, nonce, cumulative_amount, signature}  (optional if free_tier)

2. Incentive plugin intercepts
   - If X-Mesh-Payment present:
     - Verify signature, check channel balance, check cumulative_amount
     - Mark request as PAID → mines at 1.0x weight
   - If missing AND free_tier enabled:
     - Allow request → mines at 0.2x weight
   - If missing AND free_tier disabled:
     - Reject with 402 Payment Required

3. llama-server processes inference, returns response

4. Plugin meters response: GPU-seconds consumed, tokens generated
   - Records: {consumer, gpu_seconds, paid_or_free, timestamp}
   - Adds to current epoch accumulator
   - For paid requests: payment channel signature stored as work proof

5. If paid request: next request must increment cumulative_amount by >= cost

6. At epoch close: plugin builds Merkle tree of all consumer work records
   - Leaves: (consumer_address, gpu_seconds, payment_signature_or_empty)
   - Root stored locally, submitted on-chain during submission window

7. When accumulated reward > gas threshold:
   - Plugin submits work (Merkle roots) + claims via Elytro UserOp + paymaster
   - Batches multiple epochs into one tx
   - Also settles any mature payment channels in same UserOp batch
```

## Threat Model

GPU-seconds mining is simple and model-agnostic, but it has known attack vectors. Here's an honest assessment of each, and why the system remains viable despite them.

### Attack 1: Self-dealing (most likely)

**The attack:** Alice runs a node AND sends requests to herself from a second wallet. She co-signs her own requests. Her GPU burns real GPU-seconds on requests nobody actually wanted.

**How bad:** This is the primary attack. Alice is literally burning electricity to mine tokens — which is economically equivalent to Bitcoin mining. The cost of attack = electricity. If MESH token value > electricity cost, self-dealing is profitable.

**Mitigations:**
- **Free-tier 0.2x weight** — self-dealing without paying earns 1/5th the reward. To get full weight, Alice must actually spend MESH (payment channel), which has real economic cost.
- **Dynamic per-node cap** — `max(10%, 100%/nodeCount)` bounds reward per identity. With 20 nodes, max 10%. With 5 nodes, max 20%.
- **Economic equilibrium** — as more attackers join, each GPU-second is worth less MESH. Attack is only profitable when electricity cost < MESH value per GPU-second. This is the same equilibrium Bitcoin reaches.

**Honest assessment:** Cannot be fully prevented. The defense is economic — self-dealing costs real electricity AND earns reduced weight without payment. This is an accepted tradeoff shared by every Proof-of-Useful-Work system (Render, Akash, io.net).

### Attack 2: Colluding pairs

**The attack:** Alice and Bob agree: "I'll send garbage requests to you, you send garbage requests to me." Both burn real GPU-seconds, both earn mining rewards with "real" consumer signatures.

**How bad:** Same economics as Attack 1 but with payment channel signatures (full weight). They're paying electricity to mine AND shuffling MESH between them.

**Mitigations:**
- Same economic equilibrium — costs real compute.
- The MESH they shuffle via payment channels is a circular transfer — net economic activity is zero minus gas costs.
- **At small scale:** not worth the coordination overhead vs. just honestly serving.
- **At large scale:** visible in gossip (two nodes with suspiciously symmetric traffic patterns). Community can flag.

**Honest assessment:** Possible but not more profitable than self-dealing. The coordination cost adds friction for marginal benefit.

### Attack 3: Maximizing GPU-seconds cheaply

**The attack:** Load a deliberately slow model, send it huge prompts, maximize wall-clock time per watt. Use an old slow GPU to squeeze more "seconds" out of less hardware.

**How bad:** Moderate. A slow GPU producing garbage outputs for 10 seconds earns the same as a fast GPU producing useful outputs for 10 seconds.

**Mitigations:**
- **Free-tier 0.2x** — without payment, this earns 1/5th. To get full weight, need real paying consumers who won't pay for garbage.
- **Dynamic cap** — bounds the reward per identity.
- **The market self-corrects** — if a node is slow and produces garbage, no real consumer routes to it. Payment channel revenue is zero. Only free-tier mining at 0.2x weight.

**Honest assessment:** Viable but low-yield. The 0.2x free-tier penalty and dynamic cap mean the attacker earns a fraction of the epoch reward while burning electricity. Economically bounded.

### Attack 4: Sybil consumers

**The attack:** Create 100 wallets, use them all as "consumers" to send requests to your own node with payment channels. This creates the appearance of diverse paying demand.

**How bad:** This is the strongest attack. But it requires actual MESH in each sybil wallet to open payment channels. The attacker must either mine first (at 0.2x free-tier weight) or buy MESH.

**Mitigations:**
- **Real economic cost** — each sybil consumer must deposit MESH into a payment channel. The attacker is locking capital across 100 wallets.
- **Net-zero token flow** — the MESH goes from attacker's sybil wallets → attacker's node → attacker claims. The payment volume is circular. Meanwhile, honest nodes with real external consumers are accumulating non-circular payment volume.
- **Dynamic per-node cap** — even with perfect sybil gaming, max reward per identity is bounded.
- **Merkle challenge** — anyone can challenge fraudulent work proofs during the submission window.

**Honest assessment:** Expensive to execute at scale (requires real MESH capital in sybil wallets). The 0.2x free-tier weight means bootstrapping sybil capital is slow. Not eliminated, but economically costly.

### Overall Security Posture

| Attack | Difficulty | Profitability | Fully Preventable? |
|---|---|---|---|
| Self-dealing (free) | Easy | Low (0.2x weight) | No (same as Bitcoin) |
| Self-dealing (paid) | Moderate | Medium (needs MESH capital) | No |
| Colluding pairs | Moderate | Same as self-dealing + coordination | No |
| Slow GPU farming | Easy | Low (0.2x + capped) | Mostly |
| Sybil consumers | Hard | Highest but capital-intensive | No (mitigated) |

**The honest bottom line:** No Proof-of-Useful-Work system has fully solved self-dealing. The practical defense is economic:

1. **Make it cost real resources** — GPU-seconds require electricity. Can't be faked without burning power.
2. **Penalize unproven work** — free-tier at 0.2x makes self-dealing 5x less profitable than attracting real paying users.
3. **Cap the upside** — dynamic per-node limit bounds the reward for any single attacker.
4. **Require capital for full weight** — payment channel signatures prove real economic activity, and circular sybil payments cost real locked capital.
5. **Let the market find equilibrium** — attack cost approaches token value, same as Bitcoin mining economics.

## Deployment Plan

### Phase 1: Auto-wallet + Metering (no blockchain)
- Build the incentive plugin with auto Elytro wallet generation
- `mesh-llm --auto` generates wallet on first run
- Track GPU-seconds per request, paid vs free, per consumer, per epoch (off-chain)
- Build Merkle tree infrastructure for work proofs
- `mesh-llm wallet` CLI commands
- No on-chain txs yet — just wallet + metering infrastructure

### Phase 2: Mining + Payment Channels (Sepolia testnet)
- Deploy MeshToken + PaymentChannel + MeshPaymaster on Ethereum Sepolia
- GPU-seconds mining with paid/free tiers
- Payment channels: mined MESH pays for inference on other nodes
- Paymaster: zero-ETH claiming and operations
- End-to-end: join mesh → auto-wallet → mine → spend → settle → claim
- Test with 5-10 mesh nodes
- Tune parameters: free-tier weight, submission window, claim threshold

### Phase 3: Ethereum Mainnet Launch
- Deploy contracts on Ethereum mainnet
- Fair launch — no pre-mine. First epoch starts, first miners earn.
- All 21M MESH mined by node operators over time
- Paymaster funded (bootstrap ETH for gas sponsorship)
- Elytro as default wallet in mesh-llm docs + onboarding
- Mining explorer dashboard

### Phase 4: Advanced Features
- Reputation-weighted routing in mesh-llm core
- Staking from mined tokens for higher routing priority (optional, not required to mine)
- Subscription model (flat monthly rate vs. per-request)
- Wallet UI in mesh-llm web console (localhost:3131)

## File Structure

```
mesh-llm/
├── contracts/                    # Solidity contracts (Foundry)
│   ├── src/
│   │   ├── MeshToken.sol         # ERC-20 with GPU-seconds mining + halving
│   │   ├── PaymentChannel.sol    # Unidirectional payment channels
│   │   └── MeshPaymaster.sol     # ERC-4337 gas sponsor (MESH-for-gas)
│   ├── test/
│   └── foundry.toml
├── plugin-incentive/             # mesh-llm plugin (Rust)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs               # Plugin entry point + wallet generation
│       ├── metering.rs           # GPU-second tracking per request
│       ├── mining.rs             # Epoch accumulation + Merkle tree + claim logic
│       ├── channels.rs           # Payment channel state management
│       ├── settlement.rs         # On-chain settlement via Elytro + paymaster
│       ├── verification.rs       # Payment signature verification
│       └── config.rs             # Plugin configuration
└── INCENTIVE_LAYER.md            # This document
```

## Open Questions

1. **Paymaster bootstrap funding?** The paymaster needs initial ETH to sponsor gas. Options: (a) project treasury seeds it, (b) community fund, (c) paymaster earns MESH fees and sells for ETH to self-sustain.
2. **Free-tier weight tuning?** 0.2x is a starting point. Too low = discourages bootstrapping. Too high = easy to self-deal. Needs testnet data.
3. **Submission window length?** 3 days gives nodes time to submit, but delays claims. Could be 1 day for faster flow, needs testing.
4. **Claim expiry?** 365 days is generous. Shorter expiry = less on-chain storage bloat. Longer = more forgiving for casual miners.
5. **Privacy?** Payment channels reveal consumer<>provider relationships on-chain. Worth adding stealth addresses via Elytro?
6. **L2 bridge later?** Start on Ethereum mainnet for credibility. If gas becomes prohibitive for smaller operators, can bridge MESH token to an L2 later without changing the core protocol.
