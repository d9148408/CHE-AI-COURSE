---
description: Ensure LaTeX formulas in Markdown render correctly on GitHub Preview.
---

Follow these rules to ensure LaTeX math is correctly rendered by GitHub's MathJax engine:

## 1. Inline Math Spacing Rule
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

## 2. Block Math Spacing Rule
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

## 3. Supported Syntax
- Use standard LaTeX logic supported by MathJax.
- Avoid low-level TeX commands that might not be supported in a web environment.

## Validation Steps
1. Scan document for `$` symbols.
2. Check if the character immediately preceding or following `$`, if it's not a whitespace, implies a need for spacing (e.g. CJK characters, punctuation).
3. Scan document for `$$` blocks.
4. Verify blank lines exist surrounding the block.
