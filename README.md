# sys-genmon
A very pretty system monitor for xfce4-genmon

<div align='center'>
<img src='genmon_bars.png' height=108>
</div>


## Installation

1. Compile `sys-genmon.c`. You can use `./build.sh`, or do it yourself, it's just one file.
2. Create an xfce4-genmon plugin on your panel. My panel is horizontal, with size 26.
3. Set the plugin to execute `/path/to/binary -t -s -p`. You can use `--help` to see what the options do, or read the code for yourself.