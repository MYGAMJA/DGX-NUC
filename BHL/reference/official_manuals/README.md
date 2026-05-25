# Official Manuals

This directory collects vendor documentation and reference files used while debugging the Berkeley Humanoid Lite hardware stack.

## BESC

Files under `BESC/` are official STMicroelectronics documents for the B-G431B-ESC1 electronic speed controller Discovery kit.

- `BESC/b-g431b-esc1.pdf`
  - ST data brief for B-G431B-ESC1.
  - Document title: `B-G431B-ESC1 - Electronic speed controller Discovery kit for drones with STM32G431CB`.
  - Used for high-level board features and product summary.

- `BESC/dm00564746.pdf`
  - ST user manual UM2516 for B-G431B-ESC1.
  - Revision: UM2516 Rev 4, March 2021.
  - Document title: `Electronic speed controller Discovery kit for drones with STM32G431CB - User manual`.
  - Used for board layout, connector descriptions, CAN termination behavior, and STM32G431CB pinout.
  - Note: this newer revision removed the schematic diagrams.

- `BESC/B-G431B-ESC1.pdf`
  - ST user manual UM2516 for B-G431B-ESC1.
  - Revision: UM2516 Rev 1, April 2019.
  - This is an older manual revision that includes the schematic diagrams.
  - Useful for tracing board-level details that are absent from Rev 4, including LED nets such as `LED_STLINK`, `STATUS`, and the main-board `D4 LED RED` status LED.

Official ST product page:

- https://www.st.com/en/evaluation-tools/b-g431b-esc1.html

## CANable

`canable/` is a Git submodule containing Makerbase MKS CANable reference material.

- Submodule path: `BHL/official_manuals/canable`
- Source repository: https://github.com/makerbase-mks/CANable-MKS.git
- Checked revision when added: `b5c7a39f049d19f62d5897a231bfcc6c34591afb`

The submodule includes Makerbase CANable hardware files, firmware files, and user manuals for CANable variants.

## Notes

These files are reference material only. Local debug notes and wiring observations should live in the surrounding `BHL/` markdown notes rather than being mixed into this vendor documentation folder.
