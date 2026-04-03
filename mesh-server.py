#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request, os, json, subprocess

os.chdir('/Users/jiajunzeng')

MESH_TOKEN = '0x0A773654184E5405ef9AB153159185e247118668'
RPC = 'https://ethereum-sepolia-rpc.publicnode.com'
STATE_FILE = os.path.expanduser('~/.mesh-llm/mine-state.json')
RECEIPTS_FILE = os.path.expanduser('~/.mesh-llm/receipts.json')

def cast_call(sig, *args):
    try:
        cmd = ['cast', 'call', MESH_TOKEN, sig] + list(args) + ['--rpc-url', RPC]
        r = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        return r.split(' ')[0]
    except: return '0'

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/mining':
            try:
                state = {}
                if os.path.exists(STATE_FILE):
                    with open(STATE_FILE) as f: state = json.load(f)
                receipts = []
                if os.path.exists(RECEIPTS_FILE):
                    with open(RECEIPTS_FILE) as f: receipts = json.load(f)

                addr = state.get('wallet', {}).get('address', '')
                balance = int(cast_call('balanceOf(address)(uint256)', addr)) if addr else 0
                total_minted = int(cast_call('totalMinted()(uint256)'))
                mintable = int(cast_call('mintableSupply()(uint256)'))

                data = json.dumps({
                    'wallet': addr,
                    'alias': state.get('wallet', {}).get('alias', ''),
                    'balance': balance,
                    'totalMinted': total_minted,
                    'mintable': mintable,
                    'maxSupply': 21000000 * 10**18,
                    'pendingReceipts': len(receipts),
                    'totalClaimed': state.get('totalClaimed', 0),
                    'contract': MESH_TOKEN,
                }).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        elif self.path.startswith('/api/'):
            try:
                with urllib.request.urlopen(f'http://localhost:3131{self.path}', timeout=5) as resp:
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

    def log_message(self, *a): pass

HTTPServer(('127.0.0.1', 8899), Handler).serve_forever()
