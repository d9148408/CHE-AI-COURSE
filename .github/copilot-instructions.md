# GitHub Copilot Project Instructions

## Rules for This Project
- 嚴格遵守: 注意LLM模型文字生成長度限制(Output:15K), 逐段生成同時自檢, 逐段完成後再整體檢視
- 編輯ipynb檔案時, 詳閱並遵守: .github/copilot-instructions-ipynb.md 檔案中的規範

## 0. Default Python Interpreter
Use the following conda enviroment for Python scripts
"C:\Users\Yao-ChenChuang\miniconda3\envs\PY310\python.exe"

## 1. LaTeX Math Formatting

### 1.1 Inline Math Spacing Rule
**Rule**: Always add a space between inline math delimiters (`$`) and surrounding text, especially for Chinese characters and punctuation.

**Incorrect**:
- `文字$x$文字`
- `公式：$y=ax+b$`
- `(如$R^2$)`

**Correct (Fixed)**:
- `文字 $x$ 文字`
- `公式： $y=ax+b$`
- `(如 $R^2$ )`
- *Note: Ensure spaces are added even after full-width punctuation like `：`, `，`, `？`, `（`, `）`.*

### 1.2 Block Math Spacing Rule
**Rule**: Always ensure there is an empty line before and after a display math block (`$$`).

**Incorrect**:
```markdown
Text explanation:
$$
E = mc^2
$$
Next paragraph.
```

**Correct (Fixed)**:
```markdown
Text explanation:

$$
E = mc^2
$$

Next paragraph.
```

### 1.3 Supported Syntax
- Use standard LaTeX logic supported by MathJax.
- Avoid low-level TeX commands that might not be supported in a web environment.

### Validation Steps
1. Scan document for `$` symbols.
2. Check if the character immediately preceding or following `$`, if it's not a whitespace, implies a need for spacing (e.g. CJK characters, punctuation).
3. Scan document for `$$` blocks.
4. Verify blank lines exist surrounding the block.

---

## 2. Matplotlib Label/Title Language

- Strictly avoid using Chinese characters in Matplotlib label and title parameters.
- Automatically translate any intended Chinese text into English, or derive appropriate English labels based on variable names and code logic.
- All plot labels and titles must be in English for compatibility and clarity.

---

## 3. Text Processing Protocol: Large Context & Token Management

### 0. Technical Prerequisite: Encoding & Character Integrity
- All text processing, code generation, and file outputs must use UTF-8 encoding.
- Preserve all Traditional Chinese (繁體中文) characters. Do not use ASCII-only or other encodings unless explicitly requested.

### 1. Assessment & Safety Check
- If the input text fits within a single response, process it immediately.
- If the text risks truncation or exceeds limits, STOP and use the Segmentation Strategy.

### 2. Segmentation Strategy
- Propose a logical breakdown (by chapters, sections, or blocks) to ensure context retention.
- Process the text in strict segments, not all at once.

### 3. Iterative Execution & Self-Verification
- For each segment: perform the requested modifications and immediately review for accuracy and completeness.
- Wait for user confirmation (or state "Part X of Y complete") before generating the next part.

### 4. Final Global Review
- After all segments, review the entire output for logical flow and consistency across boundaries. Edit transitions if needed.
