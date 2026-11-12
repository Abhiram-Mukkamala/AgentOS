const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const http = require('http');

let serverTerminal = null;

function activate(context) {
    console.log('AgentOS Extension Activated');

    const getPort = () => vscode.workspace.getConfiguration('agentos').get('backendPort') || 8080;
    const autoStart = vscode.workspace.getConfiguration('agentos').get('autoStartBackend');

    if (autoStart) {
        startBackend(getPort());
    }

    // Register Webview View Provider
    const provider = new AgentOSChatProvider(context.extensionUri, getPort());
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('agentos.chatView', provider)
    );

    // Dynamically update backendPort when VS Code settings change
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('agentos.backendPort')) {
                provider.updatePort(getPort());
            }
        })
    );

    // Register Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('agentos.startServer', () => {
            startBackend(getPort());
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('agentos.stopServer', () => {
            stopBackend();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('agentos.openSettings', () => {
            vscode.commands.executeCommand('workbench.action.openSettings', 'agentos');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('agentos.focus', () => {
            vscode.commands.executeCommand('agentos.chatView.focus');
        })
    );
}

function startBackend(port) {
    if (serverTerminal) {
        vscode.window.showInformationMessage('AgentOS backend terminal is already running.');
        return;
    }

    const workspaceFolder = vscode.workspace.workspaceFolders ? vscode.workspace.workspaceFolders[0].uri.fsPath : '';
    if (!workspaceFolder) {
        console.warn('AgentOS: No workspace folder is open. Backend server not started.');
        return;
    }

    let serverCwd = workspaceFolder;
    const agentosV2Path = path.normalize(path.join(workspaceFolder, 'agentos-v2'));
    if (agentosV2Path.toLowerCase().startsWith(workspaceFolder.toLowerCase()) && fs.existsSync(agentosV2Path)) {
        serverCwd = agentosV2Path;
    }

    vscode.window.showInformationMessage(`Starting AgentOS backend on port ${port}...`);

    serverTerminal = vscode.window.createTerminal({
        name: 'AgentOS Backend',
        cwd: serverCwd,
        env: {
            PORT: port.toString()
        }
    });

    serverTerminal.sendText('python main.py');
    serverTerminal.show();
}

function stopBackend() {
    if (serverTerminal) {
        serverTerminal.dispose();
        serverTerminal = null;
        vscode.window.showInformationMessage('AgentOS backend stopped.');
    } else {
        vscode.window.showInformationMessage('AgentOS backend is not running.');
    }
}

class AgentOSChatProvider {
    constructor(extensionUri, port) {
        this._extensionUri = extensionUri;
        this._port = port;
        this._webviewView = null;
    }

    updatePort(newPort) {
        this._port = newPort;
        if (this._webviewView) {
            this._webviewView.webview.postMessage({
                type: 'portUpdated',
                port: this._port
            });
        }
    }

    resolveWebviewView(webviewView, context, _token) {
        this._webviewView = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage': {
                    try {
                        const response = await this._postToBackend('/chat', {
                            prompt: data.value,
                            thread_id: data.threadId || undefined
                        });
                        webviewView.webview.postMessage({
                            type: 'addResponse',
                            value: response.response,
                            threadId: response.thread_id
                        });
                    } catch (err) {
                        webviewView.webview.postMessage({
                            type: 'addError',
                            value: `Failed to connect to AgentOS backend (127.0.0.1:${this._port}): ${err.message}`
                        });
                    }
                    break;
                }
                case 'openFile': {
                    const filePath = data.path;
                    const workspaceFolder = vscode.workspace.workspaceFolders ? vscode.workspace.workspaceFolders[0].uri.fsPath : '';
                    const fullPath = path.isAbsolute(filePath) ? filePath : path.join(workspaceFolder, filePath);
                    try {
                        const doc = await vscode.workspace.openTextDocument(fullPath);
                        await vscode.window.showTextDocument(doc);
                    } catch (e) {
                        vscode.window.showErrorMessage(`Failed to open file: ${e.message}`);
                    }
                    break;
                }
                case 'insertCode': {
                    const editor = vscode.window.activeTextEditor;
                    if (editor) {
                        editor.edit(editBuilder => {
                            editBuilder.insert(editor.selection.active, data.value);
                        });
                        vscode.window.showInformationMessage('Code inserted at cursor.');
                    } else {
                        vscode.window.showErrorMessage('No active text editor found. Please click inside an editor first.');
                    }
                    break;
                }
                case 'createFile': {
                    try {
                        const doc = await vscode.workspace.openTextDocument({
                            content: data.value,
                            language: data.language
                        });
                        await vscode.window.showTextDocument(doc);
                    } catch (e) {
                        vscode.window.showErrorMessage(`Failed to create new file: ${e.message}`);
                    }
                    break;
                }
                case 'clearSession': {
                    try {
                        await this._postToBackend('/chat', { prompt: 'reset' });
                        webviewView.webview.postMessage({ type: 'sessionCleared' });
                    } catch (err) {
                        // ignore or report
                    }
                    break;
                }
            }
        });
    }

    _postToBackend(endpoint, body) {
        return new Promise((resolve, reject) => {
            const data = JSON.stringify(body);
            const options = {
                hostname: '127.0.0.1',
                port: this._port,
                path: endpoint,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(data)
                }
            };

            const req = http.request(options, (res) => {
                let rawData = '';
                res.on('data', (chunk) => { rawData += chunk; });
                res.on('end', () => {
                    if (res.statusCode === 200) {
                        try {
                            const parsed = JSON.parse(rawData);
                            resolve(parsed);
                        } catch (e) {
                            reject(new Error('Invalid JSON response from server'));
                        }
                    } else {
                        reject(new Error(`Server returned code ${res.statusCode}`));
                    }
                });
            });

            req.on('error', (e) => {
                reject(e);
            });

            req.write(data);
            req.end();
        });
    }

    _getHtmlForWebview(webview) {
        const basePath = this._extensionUri.fsPath;
        const htmlPath = path.normalize(path.join(basePath, 'webview.html'));
        if (!htmlPath.toLowerCase().startsWith(basePath.toLowerCase())) {
            console.warn("Invalid path specified!");
        }
        let html = fs.readFileSync(htmlPath, 'utf8');

        html = html.replace(/{{PORT}}/g, String(this._port));
        return html;
    }
}

function deactivate() {
    stopBackend();
}

module.exports = {
    activate,
    deactivate
};
