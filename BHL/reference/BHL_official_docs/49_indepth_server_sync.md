<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/syncing-files-from-training-server
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/syncing-files-from-training-server.md
synced: 2026-05-24
title: 학습서버 sync
-->

# Syncing Files From Training Server

For larger training runs, we typically would use a dedicated remote training server to run.

This short document notes down a way to easily synchronize training checkpoints from the server down to local workspace.

To do this, we will use this VSCode extension:

{% embed url="<https://github.com/Natizyskunk/vscode-sftp>" %}

After download, create a `sftp.json` configuration under the `.vscode/` directory and add these configurations:

```json
{
    "name": "Your Remote Training Server",
    "host": "server.ip.address",
    "port": 22,
    "protocol": "sftp",
    "context": "./logs",
    "remotePath": "/PATH/TO/THE/WORKSPACE/logs",
    "username": "YOUR-USERNAME",
    "ignore": [],
    "uploadOnSave": false,
    "useTempFile": false,
    "openSsh": false,
    "privateKeyPath": "~/.ssh/id_ed25519"
}

```

\
Now, you can run "> SFTP: Sync Remote -> Local" command to fetch all the training logs and checkpoints.


---
