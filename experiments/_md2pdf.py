#!/usr/bin/env python3
"""Convert paper markdown to PDF using weasyprint + markdown2."""
import markdown2, weasyprint, sys, os

md_file = "paper/paper_draft_v0.2.md"
pdf_file = "paper/paper_v3.3.pdf"

with open(md_file, "r") as f:
    md = f.read()

# Convert markdown to HTML with extensions
html = markdown2.markdown(md, extras=[
    "tables", "fenced-code-blocks", "footnotes", "cuddled-lists",
    "code-friendly", "mathjax", "latex-macros", "task_list",
    "header-ids", "strike", "target-blank-links"
])

# Add basic styling
full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Disentangling Optimizer and Parameter Form</title>
<style>
  body {{ max-width: 800px; margin: 40px auto; padding: 0 20px;
         font-family: 'Latin Modern Roman', 'Georgia', 'Times New Roman', serif;
         font-size: 11pt; line-height: 1.5; color: #111; }}
  h1 {{ font-size: 20pt; text-align: center; margin-bottom: 4px; }}
  h2 {{ font-size: 14pt; margin-top: 24px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  h3 {{ font-size: 12pt; margin-top: 18px; }}
  table {{ border-collapse: collapse; margin: 12px 0; font-size: 9pt; width: 100%; }}
  th, td {{ border: 1px solid #999; padding: 4px 6px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  pre, code {{ font-family: 'Latin Modern Mono', 'Courier New', monospace; font-size: 9pt; }}
  pre {{ background: #f8f8f8; padding: 8px; border-radius: 4px; overflow-x: auto; }}
  p {{ margin: 6px 0; text-align: justify; }}
  strong {{ color: #222; }}
  .math {{ font-style: italic; }}
</style></head><body>
{html}
</body></html>"""

weasyprint.HTML(string=full_html).write_pdf(pdf_file)
print(f"PDF generated: {pdf_file} ({os.path.getsize(pdf_file) // 1024} KB)")
