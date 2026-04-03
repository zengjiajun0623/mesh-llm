#!/usr/bin/env node
/**
 * mesh-client — Use mesh-llm inference and auto-sign receipts
 *
 * Wraps the mesh-llm OpenAI-compatible API. After each response,
 * automatically signs a receipt with your Elytro wallet and sends
 * it to the node operator so they can mint MESH.
 *
 * Usage:
 *   node mesh-client.js "What is quantum computing?"
 *   node mesh-client.js --model DeepSeek-R1-Distill-Qwen-32B-Q4_K_M "Explain RSA"
 *   echo "Summarize this" | node mesh-client.js
 *   node mesh-client.js --setup    # create Elytro wallet
 */

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const crypto = require("crypto");

// ── Config ────────────────────────────────────────────────────────────
const MESH_API = process.env.MESH_API || "http://localhost:9337";
const RECEIPT_URL = process.env.RECEIPT_URL || "http://localhost:9338";
const MESH_DIR = path.join(process.env.HOME, ".mesh-llm");
const CLIENT_KEY_FILE = path.join(MESH_DIR, "client-key.json");

// ── Key management ────────────────────────────────────────────────────
function ensureKey() {
  if (fs.existsSync(CLIENT_KEY_FILE)) {
    return JSON.parse(fs.readFileSync(CLIENT_KEY_FILE, "utf-8"));
  }

  // Try to use existing Elytro wallet
  try {
    const list = JSON.parse(execSync("elytro account list", { encoding: "utf-8" }));
    const acct = list.result?.accounts?.find(a => a.chainId === 11155111 && a.deployed);
    if (acct) {
      // Elytro is a smart account — we need an EOA signer for receipts
      // Generate a dedicated signing key for this client
    }
  } catch {}

  // Generate a new signing key
  log("First run — generating receipt signing key...");
  const privateKey = "0x" + crypto.randomBytes(32).toString("hex");
  const address = execSync(`cast wallet address ${privateKey}`, { encoding: "utf-8" }).trim();

  const keyData = { privateKey, address };
  fs.mkdirSync(MESH_DIR, { recursive: true });
  fs.writeFileSync(CLIENT_KEY_FILE, JSON.stringify(keyData, null, 2));
  log(`Client key: ${address}`);
  log(`Stored at: ${CLIENT_KEY_FILE}`);
  return keyData;
}

// ── Receipt signing ───────────────────────────────────────────────────
function signReceipt(privateKey, clientAddress, nodeAddress, timestamp) {
  // receipt = keccak256(abi.encode(node, consumer, timestamp))
  const encoded = execSync(
    `cast abi-encode "f(address,address,uint256)" ${nodeAddress} ${clientAddress} ${timestamp}`,
    { encoding: "utf-8" }
  ).trim();
  const receiptHash = execSync(`cast keccak ${encoded}`, { encoding: "utf-8" }).trim();
  const signature = execSync(
    `cast wallet sign --private-key ${privateKey} ${receiptHash}`,
    { encoding: "utf-8" }
  ).trim();
  return signature;
}

function sendReceipt(nodeAddress, clientAddress, timestamp, signature) {
  return new Promise((resolve) => {
    const body = JSON.stringify({ consumer: clientAddress, timestamp, signature });
    const url = new URL(RECEIPT_URL + "/receipt");
    const opts = {
      hostname: url.hostname,
      port: url.port,
      path: "/receipt",
      method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
      timeout: 5000,
    };
    const req = http.request(opts, (res) => {
      let d = "";
      res.on("data", c => d += c);
      res.on("end", () => resolve(d));
    });
    req.on("error", () => resolve(null));
    req.write(body);
    req.end();
  });
}

// ── Get node address ──────────────────────────────────────────────────
function getNodeAddress() {
  return new Promise((resolve) => {
    http.get(RECEIPT_URL + "/receipt/info", { timeout: 3000 }, (res) => {
      let d = "";
      res.on("data", c => d += c);
      res.on("end", () => {
        try { resolve(JSON.parse(d).node); } catch { resolve(null); }
      });
    }).on("error", () => resolve(null));
  });
}

// ── Chat ──────────────────────────────────────────────────────────────
function chat(model, prompt) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: model || "",
      messages: [{ role: "user", content: prompt }],
      stream: true,
    });

    const url = new URL(MESH_API + "/v1/chat/completions");
    const opts = {
      hostname: url.hostname,
      port: url.port,
      path: "/v1/chat/completions",
      method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
      timeout: 120000,
    };

    const req = http.request(opts, (res) => {
      let fullResponse = "";

      res.on("data", (chunk) => {
        const lines = chunk.toString().split("\n");
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") continue;
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) {
              process.stdout.write(content);
              fullResponse += content;
            }
          } catch {}
        }
      });

      res.on("end", () => {
        console.log(""); // newline
        resolve(fullResponse);
      });
    });

    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

// ── Helpers ───────────────────────────────────────────────────────────
function log(msg) {
  console.error(`[mesh] ${msg}`);
}

// ── Main ──────────────────────────────────────────────────────────────
async function main() {
  const args = process.argv.slice(2);

  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
  mesh-client — Chat with mesh-llm, auto-sign receipts for MESH mining

  Usage:
    node mesh-client.js "your prompt"
    node mesh-client.js --model ModelName "your prompt"
    echo "prompt" | node mesh-client.js
    node mesh-client.js --setup

  The client automatically signs a receipt after each response,
  rewarding the node operator with 1 MESH token.

  Env:
    MESH_API=http://localhost:9337     mesh-llm API
    RECEIPT_URL=http://localhost:9338   node's receipt server
`);
    return;
  }

  // Setup
  const key = ensureKey();

  if (args.includes("--setup")) {
    log(`Client address: ${key.address}`);
    log(`Key file: ${CLIENT_KEY_FILE}`);
    log("Ready to sign receipts.");
    return;
  }

  // Parse args
  let model = "";
  let prompt = "";
  const modelIdx = args.indexOf("--model");
  if (modelIdx !== -1 && args[modelIdx + 1]) {
    model = args[modelIdx + 1];
    prompt = args.filter((_, i) => i !== modelIdx && i !== modelIdx + 1).join(" ");
  } else {
    prompt = args.join(" ");
  }

  // Read from stdin if no prompt
  if (!prompt) {
    prompt = fs.readFileSync(0, "utf-8").trim();
  }

  if (!prompt) {
    log("No prompt provided. Usage: node mesh-client.js \"your question\"");
    process.exit(1);
  }

  // Chat
  try {
    await chat(model, prompt);
  } catch (e) {
    log(`API error: ${e.message}. Is mesh-llm running on ${MESH_API}?`);
    process.exit(1);
  }

  // Sign and send receipt
  const nodeAddress = await getNodeAddress();
  if (!nodeAddress) {
    log("Node receipt server not found. Skipping receipt.");
    return;
  }

  const timestamp = Math.floor(Date.now() / 1000);
  try {
    const signature = signReceipt(key.privateKey, key.address, nodeAddress, timestamp);
    const result = await sendReceipt(nodeAddress, key.address, timestamp, signature);
    if (result) {
      const parsed = JSON.parse(result);
      if (parsed.ok) {
        log(`Receipt signed for ${nodeAddress.slice(0, 8)}... (pending: ${parsed.pending})`);
      }
    }
  } catch (e) {
    log(`Receipt error: ${e.message}`);
  }
}

main().catch(e => { console.error("Fatal:", e.message); process.exit(1); });
