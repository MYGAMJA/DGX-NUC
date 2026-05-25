<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/exporting-robot-description-files-from-onshape
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/exporting-robot-description-files-from-onshape.md
synced: 2026-05-24
title: Onshape export
-->

# Exporting Robot Description Files from Onshape

## Setting up Onshape developer key

Get API key from the Onshape Developer Portal

{% embed url="<https://cad.onshape.com/appstore/dev-portal>" %}

When creating the key, make sure that at least “Application can read your documents” is selected.

<figure><img src="/files/IRTWP0JQwOvebHXfhNUv" alt=""><figcaption></figcaption></figure>

export the key as environment variable

```bash
export ONSHAPE_API=https://cad.onshape.com
export ONSHAPE_ACCESS_KEY=Your_Access_Key
export ONSHAPE_SECRET_KEY=Your_Secret_Key
```

{% hint style="info" %}

## Note

It might be more convenient to put the above three lines into a file and source it everything we need it.

For example, we can do

```bash
source ~/Documents/onshape-api-key.sh
```

{% endhint %}

## Generating the description files

Please refer to the [README](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets) of Berkeley-Humanoid-Lite-Assets repository for more information and instructions.


---
