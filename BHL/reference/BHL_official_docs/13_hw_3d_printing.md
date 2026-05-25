<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware/3d-printing-instructions
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware/3d-printing-instructions.md
synced: 2026-05-24
title: 3D 프린팅
-->

# 3D Printing Instructions

## Files

Please refer to the Releases page for the latest release of CAD model and 3D printing project files.

{% content-ref url="/pages/ReTulzQ4ECBUQPnquXTq" %}
[Releases](/docs/releases.md)
{% endcontent-ref %}

## Print settings

The following parameters are tuned for the Bambu Lab X1C 3D Printer. Additional modifications might be required to fit your own printer's characteristics.

## Printing the actuator

### Actuator Housing Profile

For the housing, output shaft, and the motor shell, the Actuator Housing Profile should be used.

<figure><img src="/files/2VFIkkK2XUlEeXTKzfrX" alt=""><figcaption></figcaption></figure>

{% tabs %}
{% tab title="Quality" %}

<figure><img src="/files/rfbufYaAydccHvUywk6B" alt=""><figcaption></figcaption></figure>
{% endtab %}

{% tab title="Strength" %}

<figure><img src="/files/6zx1jVucFkit12BQdr2d" alt=""><figcaption></figcaption></figure>
{% endtab %}

{% tab title="Speed" %}

<figure><img src="/files/rMsP7h3jGnVmfOdO8OGW" alt=""><figcaption></figcaption></figure>
{% endtab %}

{% tab title="Support" %}

<figure><img src="/files/xRzcr2uTocV0BPAbehR7" alt=""><figcaption></figcaption></figure>
{% endtab %}

{% tab title="Others" %}

<figure><img src="/files/E3wDeMNK22H3KEi8hoEU" alt=""><figcaption></figcaption></figure>
{% endtab %}
{% endtabs %}

### Actuator Shaft Profile

For the cycloidal disk, input shaft, motor shaft, and the spacers, the Actuator Shaft Profile should be used.

<figure><img src="/files/7oExDT57NpSkpAysuVwV" alt=""><figcaption></figcaption></figure>

{% tabs %}
{% tab title="Quality" %}

<figure><img src="/files/HfS6iGyyUQAdkn41NWQp" alt=""><figcaption></figcaption></figure>
{% endtab %}

{% tab title="Strength" %}

<figure><img src="/files/z1LBd6TXQEGnIhQTgSen" alt=""><figcaption></figcaption></figure>
{% endtab %}

{% tab title="Speed" %}

<figure><img src="/files/QGKi4leW3yhm0H0MA5TT" alt=""><figcaption></figcaption></figure>
{% endtab %}

{% tab title="Support" %}

<figure><img src="/files/f9UnDc5vVQOB2rrAi5eL" alt=""><figcaption></figcaption></figure>
{% endtab %}

{% tab title="Others" %}

<figure><img src="/files/k7oJQpBXldMqrAgqZgA4" alt=""><figcaption></figcaption></figure>
{% endtab %}
{% endtabs %}

## Printing the rest of the robot

Similar principle applies to the rest of the robot.&#x20;

<figure><img src="/files/ExpHvs9Sud9hsxo3K1kP" alt=""><figcaption></figcaption></figure>

Parts on the Upper Body and Lower Body plates need to be printed twice in mirrored setting to assemble the two arms and two legs. This can be achieved by right-clicking the part and selet "mirror along X axis".

The structural parts does not require the high precision as the actuator modules, so they can be printed at a faster speed setting.


---
