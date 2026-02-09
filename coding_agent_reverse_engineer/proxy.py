#!/usr/bin/env python3
"""
Proxy server to log coding agent CLI requests before forwarding to the upstream API.

Usage:
    Claude Code: python proxy.py --mode claude
    Codex CLI:   python proxy.py --mode codex

    Then run your CLI with the proxy base URL:
        ANTHROPIC_BASE_URL=http://localhost:8000 claude
        OPENAI_BASE_URL=http://localhost:8000 codex
"""

import argparse
import json
import logging
import os
import uuid
from collections import deque
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODE_CONFIG = {
    'claude': {
        'name': 'Claude Code',
        'api_url': 'https://api.anthropic.com',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'client_env': 'ANTHROPIC_BASE_URL',
    },
    'codex': {
        'name': 'Codex CLI',
        'api_url': 'https://api.openai.com',
        'api_key_env': 'OPENAI_API_KEY',
        'client_env': 'OPENAI_BASE_URL',
    },
}

# Defaults (overridden by CLI args in __main__)
PROXY_MODE = 'claude'
TARGET_API_URL = MODE_CONFIG['claude']['api_url']
API_KEY = os.environ.get('ANTHROPIC_API_KEY')


def _apply_auth_header(headers, api_key):
    """Set the correct auth header based on proxy mode."""
    if not api_key:
        return
    if PROXY_MODE == 'codex':
        headers['Authorization'] = f'Bearer {api_key}'
    else:
        headers['x-api-key'] = api_key

# In-memory storage for requests (max 100 recent requests)
request_history = deque(maxlen=100)

# File logging setup
LOG_FILE = os.environ.get("PROXY_LOG_FILE", "logs/proxy_requests.jsonl")


def log_request(method, path, headers, body):
    """Log request details in a readable format."""
    logger.info("=" * 80)
    logger.info("📤 OUTGOING REQUEST")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Method: {method}")
    logger.info(f"Path: {path}")
    logger.info("-" * 80)
    logger.info("Headers:")
    for key, value in headers.items():
        # Mask authorization header for security
        if key.lower() == "authorization":
            logger.info(f"  {key}: {value[:20]}..." if len(value) > 20 else f"  {key}: ***")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("-" * 80)
    if body:
        try:
            # Pretty print JSON body
            body_json = json.loads(body)
            logger.info("Body:")
            logger.info(json.dumps(body_json, indent=2))
        except json.JSONDecodeError:
            logger.info(f"Body (raw): {body[:500]}...")
    logger.info("=" * 80)


def log_response(status_code, headers, body):
    """Log response details."""
    logger.info("-" * 80)
    logger.info("📥 RESPONSE")
    logger.info(f"Status: {status_code}")
    logger.info("Headers:")
    for key, value in headers.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 80)
    if body:
        try:
            body_json = json.loads(body)
            logger.info("Body:")
            logger.info(json.dumps(body_json, indent=2))
        except json.JSONDecodeError:
            logger.info(f"Body (raw): {body[:500]}...")
    logger.info("=" * 80 + "\n")


def save_request(
    method,
    path,
    headers,
    body,
    status_code=None,
    response_headers=None,
    response_body=None,
):
    """Save request and response to history."""
    request_id = str(uuid.uuid4())

    # Parse bodies if they're JSON
    parsed_request_body = None
    if body:
        try:
            parsed_request_body = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            parsed_request_body = (
                body.decode('utf-8', errors='ignore')
                if isinstance(body, bytes)
                else str(body)
            )

    parsed_response_body = None
    if response_body:
        try:
            parsed_response_body = json.loads(response_body)
        except (json.JSONDecodeError, TypeError):
            parsed_response_body = (
                response_body[:1000]
                if isinstance(response_body, str)
                else str(response_body)[:1000]
            )

    request_data = {
        'id': request_id,
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'path': path,
        'url': f"{TARGET_API_URL}/{path}",
        'request_headers': dict(headers),
        'request_body': parsed_request_body,
        'status_code': status_code,
        'response_headers': dict(response_headers) if response_headers else None,
        'response_body': parsed_response_body
    }

    request_history.appendleft(request_data)

    # Append to log file (JSONL format - one JSON object per line)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(request_data) + '\n')
    except Exception as e:
        logger.error(f"Failed to write to log file: {e}")

    return request_id


@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def proxy(path):
    """Proxy all requests to the upstream API."""

    # For codex mode, ensure /v1 prefix so both OPENAI_BASE_URL=http://localhost:8000
    # and OPENAI_BASE_URL=http://localhost:8000/v1 work
    if PROXY_MODE == 'codex' and not path.startswith('v1/') and path != 'v1':
        path = f"v1/{path}"

    target_url = f"{TARGET_API_URL}/{path}"

    headers = dict(request.headers)
    headers.pop('Host', None)
    _apply_auth_header(headers, API_KEY)

    # Get request body
    body = request.get_data()

    # Log the request
    log_request(request.method, path, headers, body)

    try:
        # Forward request to Anthropic API
        response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=body,
            stream=True,  # Stream response for SSE
            timeout=300
        )

        # Check if response is streaming (Server-Sent Events)
        content_type = response.headers.get('content-type', '')
        is_streaming = 'text/event-stream' in content_type

        if is_streaming:
            # Handle streaming response
            logger.info(f"📥 STREAMING RESPONSE (Status: {response.status_code})")

            # Save request with streaming indicator
            save_request(
                request.method,
                path,
                headers,
                body,
                response.status_code,
                dict(response.headers),
                "[Streaming response - not captured]"
            )

            def generate():
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        # Log each SSE chunk
                        chunk_str = chunk.decode('utf-8', errors='ignore')
                        logger.info(f"Stream chunk: {chunk_str[:200]}...")
                        yield chunk

            # Return streaming response
            response_headers = [(k, v) for k, v in response.headers.items()
                               if k.lower() not in ['content-length', 'transfer-encoding']]
            return Response(generate(), status=response.status_code, headers=response_headers)
        else:
            # Handle regular response
            response_body = response.content

            # Log the response
            log_response(
                response.status_code,
                dict(response.headers),
                response_body.decode('utf-8', errors='ignore') if response_body else None
            )

            # Save request and response to history
            save_request(
                request.method,
                path,
                headers,
                body,
                response.status_code,
                dict(response.headers),
                response_body.decode('utf-8', errors='ignore') if response_body else None
            )

            # Return response
            response_headers = [(k, v) for k, v in response.headers.items()
                               if k.lower() not in ['content-length', 'transfer-encoding']]
            return Response(response_body, status=response.status_code, headers=response_headers)

    except Exception as e:
        logger.error(f"❌ Error forwarding request: {e}")
        return {"error": str(e)}, 500


@app.route('/')
def index():
    """Health check endpoint."""
    cfg = MODE_CONFIG[PROXY_MODE]
    return {
        "status": "running",
        "mode": PROXY_MODE,
        "message": f"{cfg['name']} Proxy Server",
        "api_key_set": bool(API_KEY),
        "dashboard_url": "http://localhost:8000/dashboard"
    }


@app.route('/dashboard')
def dashboard():
    """Web UI for viewing requests."""
    html = '''
<!DOCTYPE html>
<html>
<head>
    <title>API Request Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 24px; margin-bottom: 5px; }
        .header p { color: #666; font-size: 14px; }
        .container {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 20px;
            height: calc(100vh - 140px);
        }
        .request-list {
            background: white;
            border-radius: 8px;
            overflow-y: auto;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .request-item {
            padding: 12px 16px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
            transition: background 0.2s;
        }
        .request-item:hover { background: #f9f9f9; }
        .request-item.active { background: #e3f2fd; border-left: 3px solid #2196F3; }
        .request-method {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-right: 8px;
        }
        .method-POST { background: #4CAF50; color: white; }
        .method-GET { background: #2196F3; color: white; }
        .method-PUT { background: #FF9800; color: white; }
        .method-DELETE { background: #f44336; color: white; }
        .request-path {
            font-size: 13px;
            color: #333;
            margin-bottom: 4px;
            word-break: break-all;
        }
        .request-time {
            font-size: 11px;
            color: #999;
        }
        .request-status {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            margin-left: 8px;
        }
        .status-2xx { background: #4CAF50; color: white; }
        .status-4xx { background: #FF9800; color: white; }
        .status-5xx { background: #f44336; color: white; }
        .request-detail {
            background: white;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .detail-section {
            margin-bottom: 24px;
        }
        .detail-section h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .detail-content {
            background: #f9f9f9;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid #2196F3;
        }
        pre {
            margin: 0;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state h3 { margin-bottom: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 API Request Viewer</h1>
        <p>Viewing recent proxy requests • <span id="count">0</span> requests captured</p>
    </div>

    <div class="container">
        <div class="request-list" id="requestList">
            <div class="empty-state">
                <h3>No requests yet</h3>
                <p>Requests will appear here as they come through the proxy</p>
            </div>
        </div>

        <div class="request-detail" id="requestDetail">
            <div class="empty-state">
                <h3>Select a request</h3>
                <p>Click on a request from the list to view details</p>
            </div>
        </div>
    </div>

    <script>
        let currentRequestId = null;

        function formatTimestamp(isoString) {
            const date = new Date(isoString);
            return date.toLocaleTimeString() + ' ' + date.toLocaleDateString();
        }

        function getStatusClass(status) {
            if (!status) return '';
            if (status >= 200 && status < 300) return 'status-2xx';
            if (status >= 400 && status < 500) return 'status-4xx';
            if (status >= 500) return 'status-5xx';
            return '';
        }

        function renderRequestList(requests) {
            const listEl = document.getElementById('requestList');
            const countEl = document.getElementById('count');

            if (requests.length === 0) {
                listEl.innerHTML = `
                    <div class="empty-state">
                        <h3>No requests yet</h3>
                        <p>Requests will appear here as they come through the proxy</p>
                    </div>
                `;
                countEl.textContent = '0';
                return;
            }

            countEl.textContent = requests.length;

            listEl.innerHTML = requests.map(req => {
                const statusBadge = req.status_code
                    ? `<span class="request-status ${getStatusClass(req.status_code)}">
                           ${req.status_code}
                       </span>`
                    : '';
                return `
                    <div class="request-item ${req.id === currentRequestId ? 'active' : ''}"
                         onclick="selectRequest('${req.id}')">
                        <div>
                            <span class="request-method method-${req.method}">
                                ${req.method}
                            </span>
                            ${statusBadge}
                        </div>
                        <div class="request-path">${req.path}</div>
                        <div class="request-time">${formatTimestamp(req.timestamp)}</div>
                    </div>
                `;
            }).join('');
        }

        function selectRequest(id) {
            currentRequestId = id;
            fetch('/api/requests/' + id)
                .then(r => r.json())
                .then(req => {
                    renderRequestDetail(req);
                    renderRequestList(window.cachedRequests);
                });
        }

        function renderRequestDetail(req) {
            const detailEl = document.getElementById('requestDetail');

            const sections = [];

            sections.push(`
                <div class="detail-section">
                    <h3>Request Info</h3>
                    <div class="detail-content">
                        <strong>Method:</strong> ${req.method}<br>
                        <strong>Path:</strong> ${req.path}<br>
                        <strong>URL:</strong> ${req.url}<br>
                        <strong>Timestamp:</strong> ${formatTimestamp(req.timestamp)}<br>
                        ${req.status_code ? `<strong>Status:</strong> ${req.status_code}` : ''}
                    </div>
                </div>
            `);

            if (req.request_headers) {
                sections.push(`
                    <div class="detail-section">
                        <h3>Request Headers</h3>
                        <div class="detail-content">
                            <pre>${JSON.stringify(req.request_headers, null, 2)}</pre>
                        </div>
                    </div>
                `);
            }

            if (req.request_body) {
                const bodyContent = typeof req.request_body === 'object'
                    ? JSON.stringify(req.request_body, null, 2)
                    : req.request_body;
                sections.push(`
                    <div class="detail-section">
                        <h3>Request Body</h3>
                        <div class="detail-content">
                            <pre>${bodyContent}</pre>
                        </div>
                    </div>
                `);
            }

            if (req.response_headers) {
                sections.push(`
                    <div class="detail-section">
                        <h3>Response Headers</h3>
                        <div class="detail-content">
                            <pre>${JSON.stringify(req.response_headers, null, 2)}</pre>
                        </div>
                    </div>
                `);
            }

            if (req.response_body) {
                const bodyContent = typeof req.response_body === 'object'
                    ? JSON.stringify(req.response_body, null, 2)
                    : req.response_body;
                sections.push(`
                    <div class="detail-section">
                        <h3>Response Body</h3>
                        <div class="detail-content">
                            <pre>${bodyContent}</pre>
                        </div>
                    </div>
                `);
            }

            detailEl.innerHTML = sections.join('');
        }

        function loadRequests() {
            fetch('/api/requests')
                .then(r => r.json())
                .then(requests => {
                    window.cachedRequests = requests;
                    renderRequestList(requests);
                });
        }

        // Load requests initially and refresh every 2 seconds
        loadRequests();
        setInterval(loadRequests, 2000);
    </script>
</body>
</html>
    '''
    return render_template_string(html)


@app.route('/api/requests')
def api_requests():
    """Get list of all requests."""
    return jsonify(list(request_history))


@app.route('/api/requests/<request_id>')
def api_request_detail(request_id):
    """Get details of a specific request."""
    for req in request_history:
        if req['id'] == request_id:
            return jsonify(req)
    return {"error": "Request not found"}, 404


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proxy server for coding agent CLIs')
    parser.add_argument('--mode', choices=MODE_CONFIG.keys(), default='claude',
                        help='Which CLI to proxy for (default: claude)')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    PROXY_MODE = args.mode
    cfg = MODE_CONFIG[PROXY_MODE]
    TARGET_API_URL = cfg['api_url']
    API_KEY = os.environ.get(cfg['api_key_env'])

    if not API_KEY:
        logger.warning(f"⚠️  {cfg['api_key_env']} not set in environment!")

    logger.info(f"🚀 Starting {cfg['name']} Proxy Server (mode={PROXY_MODE})")
    logger.info(f"Forwarding to: {TARGET_API_URL}")
    logger.info(f"Configure with: {cfg['client_env']}=http://localhost:{args.port} ...")
    logger.info(f"📊 Dashboard: http://localhost:{args.port}/dashboard")
    logger.info(f"💾 Logging to file: {LOG_FILE}")

    app.run(host='127.0.0.1', port=args.port, debug=False)
