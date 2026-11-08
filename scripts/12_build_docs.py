"""Render paper/paper.md -> docs/paper.pdf (+ html) and docs/protocol.md -> docs/protocol.html.

pandoc (Markdown -> HTML) then PyMuPDF's Story layout engine (HTML -> PDF);
no LaTeX needed.
"""
import re
import subprocess
from pathlib import Path

import pymupdf as fitz

CSS = """
body { font-family: serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 15pt; margin-bottom: 2pt; } h2 { font-size: 12pt; margin-top: 10pt; } h3 { font-size: 10.5pt; }
p { margin: 3pt 0; text-align: justify; } table { border-collapse: collapse; font-size: 8.5pt; }
td, th { border-bottom: 0.5px solid #999; padding: 1.5pt 4pt; } img { width: 100%; }
code { font-family: monospace; font-size: 8.5pt; } .author { font-style: italic; }
"""
WEB_CSS = """<style>body{max-width:820px;margin:0 auto;padding:16px;font:16px/1.55 system-ui,sans-serif;
background:#fafaf7;color:#1d1f1c}table{border-collapse:collapse}td,th{border-bottom:1px solid #ccc;padding:3px 8px}
img{max-width:100%}@media (prefers-color-scheme: dark){body{background:#161815;color:#e8eae4}a{color:#7cc4a0}}</style>"""


def pandoc(src: Path, extra=()) -> str:
    return subprocess.run(["pandoc", str(src), "-f", "markdown", "-t", "html5", *extra],
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout


def html_to_pdf(html: str, out: Path, archive_dir: Path) -> int:
    story = fitz.Story(html=html, user_css=CSS, archive=fitz.Archive(str(archive_dir)))
    writer = fitz.DocumentWriter(str(out))
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (54, 54, -54, -54)
    pages, more = 0, 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
        pages += 1
    writer.close()
    return pages


def main() -> None:
    body = pandoc(Path("paper/paper.md"))
    meta = Path("paper/paper.md").read_text(encoding="utf-8")
    title = re.search(r'^title:\s*"(.*)"', meta, re.M).group(1)
    author = re.search(r'^author:\s*"(.*)"', meta, re.M).group(1)
    date = re.search(r'^date:\s*"(.*)"', meta, re.M).group(1)
    head = f"<h1>{title}</h1><p class='author'>{author} &middot; {date}</p>"
    html = head + body
    n = html_to_pdf(html, Path("docs/paper.pdf"), Path("."))
    # Pages serves docs/ only: copy the figures next to the HTML version
    Path("docs/figures").mkdir(exist_ok=True)
    for f in Path("figures").glob("*.png"):
        (Path("docs/figures") / f.name).write_bytes(f.read_bytes())
    Path("docs/paper.html").write_text("<!doctype html><meta charset='utf-8'><meta name='viewport' "
                                       "content='width=device-width,initial-scale=1'>"
                                       f"<title>{title}</title>{WEB_CSS}{html}", encoding="utf-8")
    proto = pandoc(Path("docs/protocol.md"), ["-s", "--metadata", "lang=en"])
    proto = proto.replace("</head>", WEB_CSS + "</head>")
    Path("docs/protocol.html").write_text(proto, encoding="utf-8")
    print(f"paper.pdf: {n} pages")


if __name__ == "__main__":
    main()
