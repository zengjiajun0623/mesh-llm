#!/usr/bin/env node
/**
 * mesh-mine — Mining daemon for MESH token
 *
 * Runs alongside mesh-llm. Automatically:
 *   1. Creates an Elytro wallet on first run
 *   2. Monitors mesh-llm for inference requests
 *   3. Tracks GPU-seconds per epoch
 *   4. Submits work and claims MESH tokens
 *
 * Usage:
 *   node mesh-mine.js              # start mining
 *   node mesh-mine.js --status     # show mining status
 *   node mesh-mine.js --claim      # force claim all pending epochs
 */

const { execSync, spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const http = require("http");

// ── Config ────────────────────────────────────────────────────────────
const MESH_DIR = path.join(
  process.env.HOME || process.env.USERPROFILE,
  ".mesh-llm"
);
const STATE_FILE = path.join(MESH_DIR, "mine-state.json");
const WALLET_FILE = path.join(MESH_DIR, "wallet.json");

// Contracts (Sepolia testnet — 5-min epochs for dev)
const CONTRACTS = {
  meshToken: "0x5f74F34113AE4C47A4e3e8Bdde7BC02121B4480c", // testnet (5-min epochs)
  meshTokenProd: "0x1577264ec9Af930835bd91eAd5eE7f437189C5B2", // prod (1-day epochs)
  paymentChannel: "0xd687d099FB08B133792C7D7294F56C66CE108376",
  meshPaymaster: "0x845737B8bC345727225E4EF0E3a417CF3bDcB4f3",
};

const RPC = "https://ethereum-sepolia-rpc.publicnode.com";
const MESH_API = "http://localhost:3131/api/status";
const POLL_INTERVAL = 30_000; // 30 seconds
const ELYTRO_ACCOUNT = "mesh-dev";

// ── State ─────────────────────────────────────────────────────────────
let state = {
  wallet: null,
  currentEpoch: null,
  epochStart: null,
  epochs: {}, // epoch => { paidGpuSeconds, freeGpuSeconds, submitted, claimed }
  lastRequestCount: null,
  lastPollTime: null,
  totalMined: 0,
};

function loadState() {
  try {
    if (fs.existsSync(STATE_FILE)) {
      state = JSON.parse(fs.readFileSync(STATE_FILE, "utf-8"));
    }
  } catch (e) {
    log("Warning: could not load state, starting fresh");
  }
}

function saveState() {
  fs.mkdirSync(MESH_DIR, { recursive: true });
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function log(msg) {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

// ── Wallet ────────────────────────────────────────────────────────────
async function ensureWallet() {
  if (state.wallet) {
    log(`Wallet: ${state.wallet.address} (${state.wallet.alias})`);
    return;
  }

  // Check if wallet file exists from a previous run
  if (fs.existsSync(WALLET_FILE)) {
    state.wallet = JSON.parse(fs.readFileSync(WALLET_FILE, "utf-8"));
    log(`Wallet loaded: ${state.wallet.address}`);
    saveState();
    return;
  }

  // Check if mesh-dev account already exists in Elytro
  try {
    const result = JSON.parse(
      execSync(`elytro account list mesh-dev 2>/dev/null`, {
        encoding: "utf-8",
      })
    );
    if (result.success && result.result?.accounts?.length > 0) {
      const acct = result.result.accounts[0];
      if (acct.chainId === 11155111) {
        state.wallet = {
          alias: acct.alias,
          address: acct.address,
          chainId: acct.chainId,
        };
        fs.mkdirSync(MESH_DIR, { recursive: true });
        fs.writeFileSync(WALLET_FILE, JSON.stringify(state.wallet, null, 2));
        saveState();
        log(`Existing wallet found: ${state.wallet.address}`);
        return;
      }
    }
  } catch (e) {
    // No existing account, create one
  }

  log("First run detected — creating Elytro wallet on Sepolia...");
  try {
    const result = JSON.parse(
      execSync(`elytro account create --chain 11155111 --alias mesh-miner`, {
        encoding: "utf-8",
      })
    );
    if (result.success) {
      state.wallet = {
        alias: result.result.alias,
        address: result.result.address,
        chainId: result.result.chainId,
      };

      // Activate (deploy) the wallet
      log("Deploying wallet on-chain (gas sponsored by Elytro)...");
      const activate = JSON.parse(
        execSync(`elytro account activate ${state.wallet.alias}`, {
          encoding: "utf-8",
        })
      );
      if (activate.success) {
        log(
          `Wallet deployed! Gas: ${activate.result.gasCost} (sponsored: ${activate.result.sponsored})`
        );
      }

      fs.mkdirSync(MESH_DIR, { recursive: true });
      fs.writeFileSync(WALLET_FILE, JSON.stringify(state.wallet, null, 2));
      saveState();
      log(`Wallet created: ${state.wallet.address}`);
    }
  } catch (e) {
    log(`ERROR: Could not create wallet: ${e.message}`);
    log("Make sure elytro CLI is installed: npm install -g @elytro/cli");
    process.exit(1);
  }
}

// ── Chain queries ─────────────────────────────────────────────────────
function castCall(contract, sig) {
  try {
    const result = execSync(
      `cast call ${contract} "${sig}" --rpc-url ${RPC} 2>/dev/null`,
      { encoding: "utf-8" }
    ).trim();
    return result.split(" ")[0]; // strip the [scientific notation] suffix
  } catch (e) {
    return null;
  }
}

function castCallWithArgs(contract, sig, ...args) {
  try {
    const result = execSync(
      `cast call ${contract} "${sig}" ${args.join(" ")} --rpc-url ${RPC} 2>/dev/null`,
      { encoding: "utf-8" }
    ).trim();
    return result.split(" ")[0];
  } catch (e) {
    return null;
  }
}

function getCurrentEpoch() {
  const epoch = castCall(CONTRACTS.meshToken, "currentEpoch()(uint256)");
  return epoch ? parseInt(epoch) : null;
}

function getBalance(address) {
  const bal = castCallWithArgs(
    CONTRACTS.meshToken,
    "balanceOf(address)(uint256)",
    address
  );
  return bal ? parseInt(bal) : 0;
}

function getEpochReward(epoch) {
  const r = castCallWithArgs(
    CONTRACTS.meshToken,
    "rewardForEpoch(uint256)(uint256)",
    epoch.toString()
  );
  return r ? parseInt(r) : 0;
}

function getTotalMinted() {
  const m = castCall(CONTRACTS.meshToken, "totalMinted()(uint256)");
  return m ? parseInt(m) : 0;
}

// ── Mesh-llm monitoring ───────────────────────────────────────────────
function fetchMeshStatus() {
  return new Promise((resolve, reject) => {
    http
      .get(MESH_API, { timeout: 5000 }, (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(e);
          }
        });
      })
      .on("error", reject);
  });
}

function estimateGpuSeconds(meshStatus, timeDeltaMs) {
  // If the node is serving (llama_ready), estimate GPU utilization
  // based on request count changes and model serving time
  if (!meshStatus.llama_ready || !meshStatus.model_name) return 0;

  const myModel = meshStatus.mesh_models?.find(
    (m) => m.name === meshStatus.model_name
  );
  if (!myModel) return 0;

  const currentRequests = myModel.request_count || 0;
  const prevRequests = state.lastRequestCount || 0;
  const newRequests = Math.max(0, currentRequests - prevRequests);

  state.lastRequestCount = currentRequests;

  if (newRequests === 0) return 0;

  // Estimate GPU-seconds per request based on model size
  // Rough heuristic: ~0.5-10 seconds per request depending on model
  const modelSizeGb = myModel.size_gb || 1;
  const gpuSecsPerRequest = Math.max(0.5, Math.min(10, modelSizeGb / 4));

  return Math.round(newRequests * gpuSecsPerRequest * 1000); // scale up for contract units
}

// ── Mining operations ─────────────────────────────────────────────────
function submitWork(epoch, paidGpuSeconds, freeGpuSeconds) {
  log(
    `Submitting work for epoch ${epoch}: paid=${paidGpuSeconds} free=${freeGpuSeconds}`
  );

  try {
    const proofRoot = execSync(
      `cast keccak "epoch${epoch}_miner_${state.wallet.address}_paid${paidGpuSeconds}_free${freeGpuSeconds}"`,
      { encoding: "utf-8" }
    ).trim();

    const calldata = execSync(
      `cast calldata "submitWork(uint256,uint256,uint256,bytes32)" ${epoch} ${paidGpuSeconds} ${freeGpuSeconds} ${proofRoot}`,
      { encoding: "utf-8" }
    ).trim();

    const account = state.wallet.alias || ELYTRO_ACCOUNT;
    const result = execSync(
      `elytro tx send ${account} --tx "to:${CONTRACTS.meshToken},data:${calldata}" 2>&1`,
      { encoding: "utf-8", timeout: 120_000 }
    );

    if (result.includes('"status": "confirmed"')) {
      log(`Work submitted for epoch ${epoch}!`);
      const txMatch = result.match(/"transactionHash":\s*"(0x[a-f0-9]+)"/);
      if (txMatch) log(`  tx: https://sepolia.etherscan.io/tx/${txMatch[1]}`);
      return true;
    } else if (result.includes('"success": false')) {
      const errMatch = result.match(/"message":\s*"([^"]+)"/);
      log(`Submit failed: ${errMatch ? errMatch[1] : "unknown error"}`);
      return false;
    }
  } catch (e) {
    log(`Submit error: ${e.message}`);
    return false;
  }
  return false;
}

function claimEpoch(epoch) {
  log(`Claiming MESH for epoch ${epoch}...`);

  try {
    const calldata = execSync(
      `cast calldata "claim(uint256)" ${epoch}`,
      { encoding: "utf-8" }
    ).trim();

    const account = state.wallet.alias || ELYTRO_ACCOUNT;
    const result = execSync(
      `elytro tx send ${account} --tx "to:${CONTRACTS.meshToken},data:${calldata}" 2>&1`,
      { encoding: "utf-8", timeout: 120_000 }
    );

    if (result.includes('"status": "confirmed"')) {
      log(`Epoch ${epoch} claimed!`);
      const txMatch = result.match(/"transactionHash":\s*"(0x[a-f0-9]+)"/);
      if (txMatch) log(`  tx: https://sepolia.etherscan.io/tx/${txMatch[1]}`);
      return true;
    } else {
      const errMatch = result.match(/"message":\s*"([^"]+)"/);
      log(`Claim failed: ${errMatch ? errMatch[1] : "unknown error"}`);
      return false;
    }
  } catch (e) {
    log(`Claim error: ${e.message}`);
    return false;
  }
}

// ── Main loop ─────────────────────────────────────────────────────────
async function pollAndMine() {
  const epoch = getCurrentEpoch();
  if (epoch === null) {
    log("Cannot reach chain — will retry");
    return;
  }

  // Epoch changed — process the previous one
  if (state.currentEpoch !== null && epoch > state.currentEpoch) {
    for (let e = state.currentEpoch; e < epoch; e++) {
      const ep = state.epochs[e];
      if (ep && !ep.submitted && (ep.paidGpuSeconds > 0 || ep.freeGpuSeconds > 0)) {
        log(`Epoch ${e} closed — submitting work...`);
        const ok = submitWork(e, ep.paidGpuSeconds, ep.freeGpuSeconds);
        if (ok) {
          ep.submitted = true;
          ep.submittedAt = Date.now();
          saveState();
        }
      }
    }
  }

  state.currentEpoch = epoch;

  // Initialize current epoch tracker
  if (!state.epochs[epoch]) {
    state.epochs[epoch] = {
      paidGpuSeconds: 0,
      freeGpuSeconds: 0,
      submitted: false,
      claimed: false,
    };
  }

  // Poll mesh-llm for inference activity
  try {
    const meshStatus = await fetchMeshStatus();
    const now = Date.now();
    const timeDelta = state.lastPollTime ? now - state.lastPollTime : POLL_INTERVAL;
    state.lastPollTime = now;

    const gpuSecs = estimateGpuSeconds(meshStatus, timeDelta);
    if (gpuSecs > 0) {
      // For now all requests are free-tier (no payment channels yet)
      state.epochs[epoch].freeGpuSeconds += gpuSecs;
      log(
        `+${gpuSecs} GPU-seconds (epoch ${epoch} total: ${state.epochs[epoch].freeGpuSeconds} free)`
      );
      saveState();
    }
  } catch (e) {
    // mesh-llm not running or not reachable — that's ok, just no GPU-seconds this tick
  }

  // Try to claim submitted epochs whose submission window has passed
  for (const [e, ep] of Object.entries(state.epochs)) {
    const epochNum = parseInt(e);
    if (ep.submitted && !ep.claimed) {
      // Check if submission window has passed (epoch + 2 windows for safety)
      if (epoch > epochNum + 2) {
        const ok = claimEpoch(epochNum);
        if (ok) {
          ep.claimed = true;
          ep.claimedAt = Date.now();
          saveState();
        }
      }
    }
  }
}

// ── Status display ────────────────────────────────────────────────────
async function showStatus() {
  loadState();

  console.log("\n  MESH Mining Status");
  console.log("  ==================\n");

  if (!state.wallet) {
    console.log("  Wallet: not initialized (run `node mesh-mine.js` first)\n");
    return;
  }

  console.log(`  Wallet:  ${state.wallet.address} (${state.wallet.alias})`);

  const balance = getBalance(state.wallet.address);
  console.log(`  Balance: ${(balance / 1e8).toFixed(2)} MESH`);

  const epoch = getCurrentEpoch();
  console.log(`  Epoch:   ${epoch}`);

  const reward = getEpochReward(epoch || 0);
  console.log(`  Reward:  ${(reward / 1e8).toFixed(2)} MESH/epoch`);

  const totalMinted = getTotalMinted();
  console.log(
    `  Minted:  ${(totalMinted / 1e8).toFixed(2)} / 21,000,000 MESH`
  );

  // Show epoch history
  const epochs = Object.entries(state.epochs || {}).sort(
    ([a], [b]) => parseInt(b) - parseInt(a)
  );
  if (epochs.length > 0) {
    console.log("\n  Recent epochs:");
    for (const [e, ep] of epochs.slice(0, 10)) {
      const status = ep.claimed
        ? "claimed"
        : ep.submitted
          ? "submitted (pending claim)"
          : "accumulating";
      const work =
        ep.paidGpuSeconds + Math.round(ep.freeGpuSeconds * 0.2);
      console.log(`    Epoch ${e}: ${work} effective GPU-sec [${status}]`);
    }
  }

  // Check mesh-llm
  try {
    const mesh = await fetchMeshStatus();
    console.log(`\n  mesh-llm: ${mesh.node_status} (${mesh.model_name || "no model"})`);
    console.log(`  Peers:    ${mesh.peers?.length || 0}`);
  } catch {
    console.log("\n  mesh-llm: not running");
  }

  console.log("");
}

// ── Force claim ───────────────────────────────────────────────────────
function forceClaim() {
  loadState();
  if (!state.wallet) {
    log("No wallet — run `node mesh-mine.js` first");
    process.exit(1);
  }

  let claimed = 0;
  for (const [e, ep] of Object.entries(state.epochs)) {
    if (ep.submitted && !ep.claimed) {
      const ok = claimEpoch(parseInt(e));
      if (ok) {
        ep.claimed = true;
        ep.claimedAt = Date.now();
        claimed++;
      }
    }
  }
  saveState();

  if (claimed > 0) {
    const bal = getBalance(state.wallet.address);
    log(`Claimed ${claimed} epoch(s). Balance: ${(bal / 1e8).toFixed(2)} MESH`);
  } else {
    log("No epochs to claim (submit work first, then wait for submission window)");
  }
}

// ── Entry point ───────────────────────────────────────────────────────
async function main() {
  const args = process.argv.slice(2);

  if (args.includes("--status") || args.includes("status")) {
    await showStatus();
    return;
  }

  if (args.includes("--claim") || args.includes("claim")) {
    forceClaim();
    return;
  }

  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
  mesh-mine — MESH token mining daemon

  Usage:
    node mesh-mine.js              Start mining (runs alongside mesh-llm)
    node mesh-mine.js --status     Show mining status and balance
    node mesh-mine.js --claim      Force claim all pending epochs

  The daemon:
    1. Creates an Elytro wallet on first run (gas sponsored)
    2. Monitors mesh-llm at localhost:3131 for inference requests
    3. Tracks GPU-seconds per epoch
    4. Auto-submits work when epochs close
    5. Auto-claims MESH tokens when submission window passes

  Requirements:
    - mesh-llm running (mesh-llm --auto)
    - Elytro CLI (npm install -g @elytro/cli)
    - Foundry cast (for chain queries)

  Config:
    State stored at ~/.mesh-llm/mine-state.json
    Wallet stored at ~/.mesh-llm/wallet.json

  Contracts (Sepolia testnet):
    MeshToken: ${CONTRACTS.meshToken}
    PaymentChannel: ${CONTRACTS.paymentChannel}
`);
    return;
  }

  // ── Start mining daemon ───────────────────────────────────────────
  console.log("");
  console.log("  ⛏  MESH Mining Daemon");
  console.log("  =====================");
  console.log("");

  loadState();
  await ensureWallet();

  const balance = getBalance(state.wallet.address);
  const epoch = getCurrentEpoch();
  const reward = getEpochReward(epoch || 0);

  console.log("");
  console.log(`  Wallet:   ${state.wallet.address}`);
  console.log(`  Balance:  ${(balance / 1e8).toFixed(2)} MESH`);
  console.log(`  Epoch:    ${epoch} | Reward: ${(reward / 1e8).toFixed(2)} MESH/epoch`);
  console.log(`  Token:    ${CONTRACTS.meshToken}`);
  console.log(`  Chain:    Sepolia (testnet)`);
  console.log("");
  console.log("  Monitoring mesh-llm at localhost:3131...");
  console.log("  Press Ctrl+C to stop.\n");

  // Initial poll
  await pollAndMine();

  // Main polling loop
  const interval = setInterval(async () => {
    try {
      await pollAndMine();
    } catch (e) {
      log(`Poll error: ${e.message}`);
    }
  }, POLL_INTERVAL);

  // Graceful shutdown
  process.on("SIGINT", () => {
    log("Shutting down...");
    clearInterval(interval);
    saveState();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    clearInterval(interval);
    saveState();
    process.exit(0);
  });
}

main().catch((e) => {
  console.error("Fatal:", e.message);
  process.exit(1);
});
