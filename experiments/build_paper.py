"""Render paper/whitepaper.md to a clean, self-contained whitepaper.html (MathJax for equations).

Pandoc is not required. If pandoc is available a PDF can also be produced with
`pandoc paper/whitepaper.md -o paper/whitepaper.pdf`, otherwise print the HTML to PDF.
"""
from __future__ import annotations
import re
import pathlib
import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "whitepaper.md"
OUT = ROOT / "paper" / "whitepaper.html"

CSS = """
:root { --ink:#1a1a1a; --muted:#666; --accent:#1f6fb2; }
* { box-sizing:border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,serif;
  max-width:780px; margin:0 auto; padding:48px 24px; color:var(--ink); line-height:1.6;
  font-size:16px; }
h1 { font-size:1.9em; line-height:1.2; margin-top:1.6em; }
h2 { font-size:1.3em; margin-top:1.8em; border-bottom:1px solid #eee; padding-bottom:4px; }
h3 { font-size:1.08em; margin-top:1.4em; color:#333; }
p, li { color:#222; }
code { background:#f4f4f6; padding:1px 5px; border-radius:4px; font-size:0.9em; }
table { border-collapse:collapse; width:100%; margin:1.2em 0; font-size:0.92em; }
th, td { border:1px solid #ddd; padding:6px 9px; text-align:left; }
th { background:#f6f7f9; }
tr td:first-child { color:var(--muted); }
strong { color:#111; }
blockquote { border-left:3px solid var(--accent); margin:1em 0; padding:2px 16px; color:#333;
  background:#f7fafc; }
.subtitle { color:var(--muted); font-size:1.05em; margin-top:-0.4em; }
.byline { color:var(--muted); margin-bottom:2em; }
"""

MATHJAX = """
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def parse_front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        text = text[m.end():]
    return meta, text


def main():
    text = SRC.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    html_body = markdown.markdown(body, extensions=["tables", "fenced_code"])
    header = ""
    if meta.get("title"):
        header += f"<h1>{meta['title']}</h1>\n"
    if meta.get("subtitle"):
        header += f"<p class='subtitle'>{meta['subtitle']}</p>\n"
    if meta.get("author"):
        header += f"<p class='byline'>{meta['author']} · {meta.get('date','')}</p>\n"
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{meta.get('title','Whitepaper')}</title>"
        f"<style>{CSS}</style>{MATHJAX}</head><body>"
        f"{header}{html_body}</body></html>"
    )
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT)
    # try pandoc PDF if available
    import shutil, subprocess
    if shutil.which("pandoc"):
        pdf = ROOT / "paper" / "whitepaper.pdf"
        try:
            subprocess.run(["pandoc", str(SRC), "-o", str(pdf)], check=True)
            print("wrote", pdf)
        except Exception as e:
            print("pandoc PDF failed:", e)
    else:
        print("pandoc not found; open whitepaper.html and print to PDF for the PDF artifact")


if __name__ == "__main__":
    main()
