import asyncio
import shutil
import sys
from pathlib import Path


def _find_soffice() -> str | None:
    if sys.platform.startswith("win"):
        windows_path = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
        if windows_path.exists():
            return str(windows_path)
        return None
    return shutil.which("soffice")


async def convert_to_pdf(src: Path, dest_dir: Path) -> Path:
    soffice = _find_soffice()
    if soffice is None:
        raise RuntimeError(
            "LibreOffice (soffice) not found. Install LibreOffice and make sure "
            "'soffice' is on PATH (Linux/macOS) or located at "
            r"'C:\Program Files\LibreOffice\program\soffice.exe' (Windows)."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)

    process = await asyncio.create_subprocess_exec(
        soffice,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(dest_dir),
        str(src),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"LibreOffice conversion timed out after 60s for {src}")

    if process.returncode != 0:
        raise RuntimeError(
            f"LibreOffice failed to convert {src} (exit code {process.returncode}): "
            f"{stderr.decode(errors='replace').strip()}"
        )

    pdf_path = dest_dir / f"{src.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(
            f"LibreOffice reported success but PDF not found at {pdf_path}. "
            f"stdout: {stdout.decode(errors='replace').strip()}"
        )

    return pdf_path
