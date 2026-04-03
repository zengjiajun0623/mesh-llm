#!/usr/bin/env node
/**
 * mesh-mine — MESH token miner for mesh-llm nodes
 *
 * Runs alongside mesh-llm. Automatically:
 *   1. Creates an Elytro wallet on first run
 *   2. Collects consumer-signed receipts when serving inference
 *   3. Submits receipts on-chain to mint MESH
 *
 * Usage:
 *   node mesh-mine.js              # start mining
 *   node mesh-mine.js --status     # show balance + stats
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const http = require("http");
const crypto = require("crypto");

// ── Config ────────────────────────────────────────────────────────────
const MESH_DIR = path.join(process.env.HOME, ".mesh-llm");
const STATE_FILE = path.join(MESH_DIR, "mine-state.json");
const RECEIPTS_FILE = path.join(MESH_DIR, "receipts.json");
const WALLET_FILE = path.join(MESH_DIR, "wallet.json");

const MESH_TOKEN = "0x0A773654184E5405ef9AB153159185e247118668";
const RPC = "https://ethereum-sepolia-rpc.publicnode.com";
const MESH_API = "http://localhost:3131/api/status";
const RECEIPT_PORT = 9338; // receipt signing endpoint for consumers
const POLL_INTERVAL = 60_000;
const CLAIM_INTERVAL = 300_000; // try to claim every 5 min
const ELYTRO_ACCOUNT_PREFIX = "mesh-miner";

let state = { wallet: null, totalClaimed: 0, totalReceipts: 0 };
let receipts = []; // { consumer, timestamp, signature }

// ── Helpers ───────────────────────────────────────────────────────────
function log(msg) {
  console.log(`[${new Date().toISOString().slice(11, 19)}] ${msg}`);
}

function loadState() {
  try { if (fs.existsSync(STATE_FILE)) state = JSON.parse(fs.readFileSync(STATE_FILE, "utf-8")); } catch {}
  try { if (fs.existsSync(RECEIPTS_FILE)) receipts = JSON.parse(fs.readFileSync(RECEIPTS_FILE, "utf-8")); } catch {}
}

function saveState() {
  fs.mkdirSync(MESH_DIR, { recursive: true });
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  fs.writeFileSync(RECEIPTS_FILE, JSON.stringify(receipts));
}

function castCall(sig, ...args) {
  try {
    const r = execSync(`cast call ${MESH_TOKEN} "${sig}" ${args.join(" ")} --rpc-url ${RPC} 2>/dev/null`, { encoding: "utf-8" }).trim();
    return r.split(" ")[0];
  } catch { return "0"; }
}

// ── Wallet ────────────────────────────────────────────────────────────
function ensureWallet() {
  if (state.wallet) { log(`Wallet: ${state.wallet.address}`); return; }
  if (fs.existsSync(WALLET_FILE)) {
    state.wallet = JSON.parse(fs.readFileSync(WALLET_FILE, "utf-8"));
    log(`Wallet loaded: ${state.wallet.address}`);
    saveState();
    return;
  }

  // Check existing Elytro accounts on Sepolia
  try {
    const list = JSON.parse(execSync("elytro account list", { encoding: "utf-8" }));
    const sepolia = list.result?.accounts?.find(a => a.chainId === 11155111 && a.deployed);
    if (sepolia) {
      state.wallet = { alias: sepolia.alias, address: sepolia.address, chainId: 11155111 };
      fs.mkdirSync(MESH_DIR, { recursive: true });
      fs.writeFileSync(WALLET_FILE, JSON.stringify(state.wallet, null, 2));
      saveState();
      log(`Existing Sepolia wallet found: ${state.wallet.address}`);
      return;
    }
  } catch {}

  // Create new wallet
  log("Creating Elytro wallet on Sepolia...");
  try {
    const alias = `${ELYTRO_ACCOUNT_PREFIX}-${Date.now().toString(36)}`;
    const r = JSON.parse(execSync(`elytro account create --chain 11155111 --alias ${alias}`, { encoding: "utf-8" }));
    if (!r.success) throw new Error("create failed");

    log("Deploying wallet (gas sponsored by Elytro)...");
    const a = JSON.parse(execSync(`elytro account activate ${alias}`, { encoding: "utf-8" }));
    log(`Wallet deployed: ${r.result.address} (sponsored: ${a.result?.sponsored})`);

    state.wallet = { alias, address: r.result.address, chainId: 11155111 };
    fs.mkdirSync(MESH_DIR, { recursive: true });
    fs.writeFileSync(WALLET_FILE, JSON.stringify(state.wallet, null, 2));
    saveState();
  } catch (e) {
    log(`ERROR creating wallet: ${e.message}`);
    log("Install Elytro CLI: npm install -g @elytro/cli");
    process.exit(1);
  }
}

// ── Receipt server ────────────────────────────────────────────────────
// Consumers hit POST /receipt after getting inference to sign a receipt
function startReceiptServer() {
  const server = http.createServer((req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") { res.writeHead(200); res.end(); return; }

    if (req.method === "POST" && req.url === "/receipt") {
      let body = "";
      req.on("data", c => body += c);
      req.on("end", () => {
        try {
          const { consumer, timestamp, signature } = JSON.parse(body);
          if (!consumer || !timestamp || !signature) throw new Error("missing fields");

          // Basic validation
          if (consumer.toLowerCase() === state.wallet.address.toLowerCase()) {
            res.writeHead(400); res.end('{"error":"self-deal"}'); return;
          }

          receipts.push({ consumer, timestamp: Number(timestamp), signature });
          saveState();

          log(`Receipt from ${consumer.slice(0, 8)}...`);
          res.writeHead(200);
          res.end(JSON.stringify({ ok: true, pending: receipts.length }));
        } catch (e) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: e.message }));
        }
      });
      return;
    }

    if (req.method === "GET" && req.url === "/receipt/info") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        node: state.wallet?.address,
        token: MESH_TOKEN,
        chain: "sepolia",
        pendingReceipts: receipts.length,
        message: "Sign receipt: keccak256(abi.encode(node, consumer, timestamp))"
      }));
      return;
    }

    res.writeHead(404); res.end("not found");
  });

  server.listen(RECEIPT_PORT, "0.0.0.0", () => {
    log(`Receipt server on :${RECEIPT_PORT}`);
  });
}

// ── Claim ─────────────────────────────────────────────────────────────
function submitReceipts() {
  if (receipts.length === 0) return;

  const batch = receipts.slice(0, 100); // max 100 per tx
  log(`Submitting ${batch.length} receipts on-chain...`);

  try {
    // Build arrays for the contract call
    const consumers = batch.map(r => r.consumer);
    const timestamps = batch.map(r => r.timestamp);
    const sigs = batch.map(r => r.signature);

    // Encode calldata
    // claim(address[],uint256[],bytes[])
    const consumersArr = `[${consumers.join(",")}]`;
    const timestampsArr = `[${timestamps.join(",")}]`;
    const sigsArr = `[${sigs.join(",")}]`;

    const calldata = execSync(
      `cast calldata "claim(address[],uint256[],bytes[])" "${consumersArr}" "${timestampsArr}" "${sigsArr}"`,
      { encoding: "utf-8" }
    ).trim();

    const account = state.wallet.alias;
    const result = execSync(
      `elytro tx send ${account} --tx "to:${MESH_TOKEN},data:${calldata}" 2>&1`,
      { encoding: "utf-8", timeout: 120_000 }
    );

    if (result.includes('"status": "confirmed"')) {
      const txMatch = result.match(/"transactionHash":\s*"(0x[a-f0-9]+)"/);
      log(`Claimed ${batch.length} receipts!`);
      if (txMatch) log(`  tx: https://sepolia.etherscan.io/tx/${txMatch[1]}`);

      // Remove claimed receipts
      receipts = receipts.slice(batch.length);
      state.totalClaimed += batch.length;
      state.totalReceipts += batch.length;
      saveState();
    } else if (result.includes("no valid receipts")) {
      log("No valid receipts in batch (expired/duplicate/rate-limited). Clearing.");
      receipts = receipts.slice(batch.length);
      saveState();
    } else {
      const errMatch = result.match(/"message":\s*"([^"]+)"/);
      log(`Claim failed: ${errMatch ? errMatch[1] : "unknown"}`);
    }
  } catch (e) {
    log(`Claim error: ${e.message}`);
  }
}

// ── Status ────────────────────────────────────────────────────────────
async function showStatus() {
  loadState();
  console.log("\n  MESH Mining Status");
  console.log("  ==================\n");

  if (!state.wallet) { console.log("  Not initialized. Run: node mesh-mine.js\n"); return; }

  const bal = parseInt(castCall("balanceOf(address)(uint256)", state.wallet.address));
  const supply = parseInt(castCall("mintableSupply()(uint256)"));
  const totalMinted = parseInt(castCall("totalMinted()(uint256)"));

  console.log(`  Wallet:    ${state.wallet.address} (${state.wallet.alias})`);
  console.log(`  Balance:   ${(bal / 1e18).toFixed(0)} MESH`);
  console.log(`  Minted:    ${(totalMinted / 1e18).toFixed(0)} / 21,000,000 MESH`);
  console.log(`  Pending:   ${receipts.length} receipts`);
  console.log(`  Claimed:   ${state.totalClaimed} receipts total`);
  console.log(`  Token:     ${MESH_TOKEN}`);
  console.log(`  Chain:     Sepolia`);

  try {
    const mesh = await fetchJson(MESH_API);
    console.log(`\n  mesh-llm:  ${mesh.node_status} (${mesh.model_name || "no model"})`);
    console.log(`  Peers:     ${mesh.peers?.length || 0}`);
  } catch { console.log("\n  mesh-llm:  not running"); }

  console.log(`\n  Receipt endpoint: http://localhost:${RECEIPT_PORT}/receipt`);
  console.log("  Consumers POST signed receipts here after using your node.\n");
}

function fetchJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, { timeout: 5000 }, res => {
      let d = ""; res.on("data", c => d += c);
      res.on("end", () => { try { resolve(JSON.parse(d)); } catch(e) { reject(e); } });
    }).on("error", reject);
  });
}

// ── Main ──────────────────────────────────────────────────────────────
async function main() {
  const args = process.argv.slice(2);
  if (args.includes("--status") || args.includes("status")) { await showStatus(); return; }
  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
  mesh-mine — MESH token miner

  Usage:
    node mesh-mine.js           Start mining (run alongside mesh-llm)
    node mesh-mine.js --status  Show balance and stats

  How it works:
    1. Creates an Elytro wallet on first run (gas sponsored)
    2. Starts a receipt server on :${RECEIPT_PORT}
    3. Consumers sign receipts after getting inference from your node
    4. Daemon submits receipts on-chain to mint 1 MESH each

  Token: ${MESH_TOKEN} (Sepolia)
  Supply: 21,000,000 MESH | 1 MESH per receipt
`);
    return;
  }

  console.log("\n  MESH Miner");
  console.log("  ==========\n");

  loadState();
  ensureWallet();

  const bal = parseInt(castCall("balanceOf(address)(uint256)", state.wallet.address));
  console.log(`\n  Wallet:    ${state.wallet.address}`);
  console.log(`  Balance:   ${(bal / 1e18).toFixed(0)} MESH`);
  console.log(`  Pending:   ${receipts.length} receipts`);
  console.log(`  Token:     ${MESH_TOKEN}`);
  console.log(`  Receipts:  http://localhost:${RECEIPT_PORT}/receipt`);
  console.log("\n  Consumers sign receipts after using your node.");
  console.log("  Receipts auto-submitted on-chain every 5 min.\n");

  startReceiptServer();

  // Periodically submit receipts
  setInterval(() => {
    if (receipts.length > 0) submitReceipts();
  }, CLAIM_INTERVAL);

  process.on("SIGINT", () => { log("Shutting down..."); saveState(); process.exit(0); });
  process.on("SIGTERM", () => { saveState(); process.exit(0); });
}

main().catch(e => { console.error("Fatal:", e.message); process.exit(1); });
