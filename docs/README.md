# Documentation

**[Predicto-Forex-Explained.pdf](Predicto-Forex-Explained.pdf)** — a plain-English
explanation of the whole project for a reader who knows nothing about forex or
trading. Eight pages: what the market is, why trading costs decide everything,
how we guard against fooling ourselves, what we tested, what failed, what
survived, and what remains unknown. Every technical term is defined where it
first appears.

Regenerate after editing the source:

```bash
chrome --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="docs/Predicto-Forex-Explained.pdf" \
  "file:///<abs-path>/docs/project_explainer.html"
```

`project_explainer.html` is the source of truth; the PDF is a build artifact.

For the full technical record — every session, every bug, every corrected
result — see [`CLAUDE.md`](../CLAUDE.md) in the repository root.
