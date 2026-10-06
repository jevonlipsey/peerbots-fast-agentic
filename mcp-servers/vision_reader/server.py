import base64
import os
import mimetypes
from io import BytesIO
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

# init fastmcp server
mcp = FastMCP("vision_reader")

# find project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()


@mcp.tool()
def read_document(filepath: str) -> list[TextContent | ImageContent]:
    """
    reads a document from the local filesystem and extracts its contents.
    supports .pdf (extracts text), .png/.jpg/.jpeg (extracts image data for vision),
    and standard text files. it returns the extracted content as either text or image content."
    """
    if not os.path.isabs(filepath):
        filepath = str(PROJECT_ROOT / filepath)

    if not os.path.exists(filepath):
        return [
            TextContent(
                type="text",
                text=f"Error: The file '{filepath}' does not exist on the local drive.",
            )
        ]

    mime_type, _ = mimetypes.guess_type(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    try:
        # pdfs
        if ext == ".pdf" or mime_type == "application/pdf":
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if not text.strip():
                text = "[System: The PDF was parsed successfully but no extractable text was found (it may be a scanned image).]"
            return [TextContent(type="text", text=f"Extracted PDF Text:\n\n{text}")]

        # images
        elif mime_type and mime_type.startswith("image/"):
            with Image.open(filepath) as img:
                # compress
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

                # save to memory buffer as a compressed jpeg
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return [ImageContent(type="image", data=b64_data, mimeType="image/jpeg")]

        # handle generic txt files
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [TextContent(type="text", text=f"Extracted File Text:\n\n{text}")]

    except Exception as e:
        return [
            TextContent(type="text", text=f"Error reading file '{filepath}': {str(e)}")
        ]


if __name__ == "__main__":
    mcp.run()
