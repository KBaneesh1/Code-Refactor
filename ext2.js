// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');
const { spawn } = require('child_process');
const fs = require('fs');
const axios = require('axios');
// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
const {exec} = require('child_process');
const path = require('path');

/**
 * @param {vscode.ExtensionContext} context
 */


function activate(context) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "Code-Refactor" is now active!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with  registerCommand
	// The commandId parameter must match the command field in package.json

	let disposable = vscode.commands.registerCommand('Code-Refactor.myCommand',async function () {
        // Get the active text editor
        let editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor found.');
            return; // No open text editor
        }

        // Get the selected text
        let selection = editor.selection;
        let selectedText = editor.document.getText(selection);

        // Now you have the selected text, you can do whatever processing you want
        console.log('Selected Text:', selectedText);
        vscode.window.showInformationMessage('Selected Text: ' + selectedText);

        // const pythonScriptPath = context.asAbsolutePath('scripts/testing.py');

        const pythonProcess = spawn('python', ['D:\\HPE CTY\\testing.py', selectedText]);
        
        pythonProcess.stdout.on('data', async (data) => {
            // const outputFile = fs.createWriteStream('D:\\HPE CTY\\output.txt', { flags: 'w' });
            const refactoredCode = data.toString()
            
            console.log(`stdout : ${data}`);
            if(refactoredCode==null)
            {
                return;
            }
                
            const workspaceFolders = vscode.workspace.workspaceFolders;
            if (!workspaceFolders) {
                vscode.window.showErrorMessage('No workspace folder is open');
                return;
            }
    
            const workspacePath = workspaceFolders[0].uri.fsPath;
            const newFilePath = path.join(workspacePath, 'refactored_output.md');
            try {
                await fs.promises.writeFile(newFilePath, refactoredCode);
                const newFileUri = vscode.Uri.file(newFilePath);

                // Open the new Markdown file in the editor
                const originalDocument = await vscode.workspace.openTextDocument(editor.document.uri);
                await vscode.window.showTextDocument(originalDocument, vscode.ViewColumn.One);
                
                const previewDocument = await vscode.workspace.openTextDocument(newFileUri);
                // await vscode.window.showTextDocument(previewDocument, vscode.ViewColumn.Beside);
                await vscode.commands.executeCommand('markdown.showPreviewToSide', newFileUri);

                vscode.window.showInformationMessage('Refactored code opened in new Markdown file');
            } catch (err) {
                console.error(`Error writing to file: ${err}`);
                vscode.window.showErrorMessage('Failed to write refactored code to file');
            }
        });

        
        pythonProcess.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`child process exited with code ${code}`);
        });

    });

	context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
	activate,
	deactivate
}
