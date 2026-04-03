#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request, os, json, subprocess

os.chdir('/Users/jiajunzeng')

MESH_TOKEN = '0x5f74F34113AE4C47A4e3e8Bdde7BC02121B4480c'
RPC = 'https://ethereum-sepolia-rpc.publicnode.com'
STATE_FILE = os.path.expanduser('~/.mesh-llm/mine-state.json')

def cast_call(contract, sig, *args):
    try:
        cmd = ['cast', 'call', contract, sig] + list(args) + ['--rpc-url', RPC]
        r = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        return r.split(' ')[0]
    except: return '0'

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/mining':
            try:
                # Read daemon state
                state = {}
                if os.path.exists(STATE_FILE):
                    with open(STATE_FILE) as f: state = json.load(f)

                addr = state.get('wallet', {}).get('address', '')
                balance = int(cast_call(MESH_TOKEN, 'balanceOf(address)(uint256)', addr)) if addr else 0
                epoch = int(cast_call(MESH_TOKEN, 'currentEpoch()(uint256)'))
                reward = int(cast_call(MESH_TOKEN, 'rewardForEpoch(uint256)(uint256)', str(epoch)))
                total_minted = int(cast_call(MESH_TOKEN, 'totalMinted()(uint256)'))

                data = json.dumps({
                    'wallet': addr,
                    'alias': state.get('wallet', {}).get('alias', ''),
                    'balance': balance,
                    'currentEpoch': epoch,
                    'epochReward': reward,
                    'totalMinted': total_minted,
                    'maxSupply': 2100000000000000,
                    'epochs': state.get('epochs', {}),
                    'lastPollTime': state.get('lastPollTime'),
                    'contract': MESH_TOKEN,
                }).encode()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        elif self.path.startswith('/api/'):
            try:
                url = f'http://localhost:3131{self.path}'
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = resp.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(data)
            except Exception as e:
                self.send_response(502)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"error":"{e}"}}'.encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass

HTTPServer(('127.0.0.1', 8899), Handler).serve_forever()
