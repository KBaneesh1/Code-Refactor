const vscode = require('vscode');
const net = require('net');
const path = require('path');
function activate(context) {
    console.log('Congratulations, your extension "Code-Refactor" is now active!');
    
    const serverHost = 'localhost';
    const serverPort = 8000;

    function sendRequest(inputData) {
        return new Promise((resolve, reject) => {
            const client = new net.Socket();
            client.connect(serverPort, serverHost, () => {
                const message = JSON.stringify({input_data: inputData });
                const messageBuffer = Buffer.from(message, 'utf-8');
                const messageLengthBuffer = Buffer.alloc(4);
                messageLengthBuffer.writeUInt32BE(messageBuffer.length, 0);
                client.write(Buffer.concat([messageLengthBuffer, messageBuffer]));
            });

            let responseData = Buffer.alloc(0);

            client.on('data', (data) => {
                responseData = Buffer.concat([responseData, data]);
                if (responseData.length >= 4) {
                    const messageLength = responseData.readUInt32BE(0);
                    if (responseData.length >= 4 + messageLength) {
                        const messageData = responseData.slice(4, 4 + messageLength).toString('utf-8');
                        resolve(JSON.parse(messageData));
                        client.destroy();
                    }
                }
            });

            client.on('error', (err) => {
                reject(err);
            });
        });
    }

    let disposable = vscode.commands.registerCommand('Code-Refactor.myCommand', async function () {
        let editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor found.');
            return;
        }

        let selection = editor.selection;
        let selectedText = editor.document.getText(selection);

        console.log('Selected Text:', selectedText);
        vscode.window.showInformationMessage('Selected Text: ' + selectedText);

        try {
            const response = await sendRequest(selectedText);
            if (response.status === 'success') {
                const refactoredCode = response.result;
                console.log(`Refactored code: ${refactoredCode}`);

                const workspaceFolders = vscode.workspace.workspaceFolders;
                if (!workspaceFolders) {
                    vscode.window.showErrorMessage('No workspace folder is open');
                    return;
                }

                const workspacePath = workspaceFolders[0].uri.fsPath;
                const newFilePath = path.join(workspacePath, 'refactored_output.md');
                try {
                    await vscode.workspace.fs.writeFile(vscode.Uri.file(newFilePath), Buffer.from(refactoredCode));
                    const newFileUri = vscode.Uri.file(newFilePath);

                    const originalDocument = await vscode.workspace.openTextDocument(editor.document.uri);
                    await vscode.window.showTextDocument(originalDocument, vscode.ViewColumn.One);
                    
                    const previewDocument = await vscode.workspace.openTextDocument(newFileUri);
                    await vscode.commands.executeCommand('markdown.showPreviewToSide', newFileUri);

                    vscode.window.showInformationMessage('Refactored code opened in new Markdown file');
                } catch (err) {
                    console.error(`Error writing to file: ${err}`);
                    vscode.window.showErrorMessage('Failed to write refactored code to file');
                }
            } else {
                vscode.window.showErrorMessage(`Error: ${response.message}`);
            }
        } catch (err) {
            console.error(`Error: ${err}`);
            vscode.window.showErrorMessage('Failed to communicate with the server');
        }
    });

    context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
}
