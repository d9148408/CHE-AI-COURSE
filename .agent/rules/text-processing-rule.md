---
trigger: always_on
---

### Text Processing Protocol: Large Context & Token Management
0. Technical Prerequisite: Encoding & Character Integrity

Mandatory UTF-8 Encoding: All text processing, code generation, and file outputs must strictly adhere to UTF-8 encoding.

Character Preservation: Ensure full compatibility for Traditional Chinese (繁體中文) characters. Do not use ASCII-only modes or other encodings (like Big5/GBK) unless explicitly requested, to prevent garbled text (mojibake).

1. Assessment & Safety Check: Before initiating any modification task, assess the input text length relative to your maximum context window (token limit).

IF the text fits comfortably within a single response: Proceed with the modification immediately.

IF the text risks truncation or exceeds limits: STOP and initiate the Segmentation Strategy.

2. Segmentation Strategy (If triggered):

Plan: Propose a logical breakdown of the text (e.g., by chapters, sections, or thematic blocks) to ensure context retention.

Execute Sequentially: Process the text in strict segments. Do not attempt to output the entire result at once.

3. Iterative Execution & Self-Verification: For each processed segment:

Perform the requested modifications.

Self-Correction: Immediately review the output for accuracy, consistency, and completeness before moving to the next segment.

Wait for user confirmation (or explicitly state "Part X of Y complete") before generating the next part.

4. Final Global Review: Once all segments are processed, conduct a final holistic review to ensure logical flow and consistency across boundaries. Identify if any transitional edits are required.