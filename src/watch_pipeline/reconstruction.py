import asyncio
import base64
import time
from pathlib import Path

import httpx

PIAPI_BASE_URL = "https://api.piapi.ai/api/v1"


def get_cached_mesh(reference: str, cache_dir: Path) -> Path | None:
    safe_name = reference.replace("/", "_").replace(" ", "_")
    mesh_path = cache_dir / f"{safe_name}.glb"
    return mesh_path if mesh_path.exists() else None


async def reconstruct_watch(
    photo_path: Path,
    api_key: str,
    cache_dir: Path,
    reference: str,
    poll_interval: float = 5.0,
    timeout: float = 120.0,
) -> Path:
    cached = get_cached_mesh(reference, cache_dir)
    if cached is not None:
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    image_b64 = base64.b64encode(photo_path.read_bytes()).decode()

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Submit reconstruction task
        resp = await client.post(
            f"{PIAPI_BASE_URL}/task",
            headers={"X-API-Key": api_key},
            json={
                "model": "trellis",
                "task_type": "image-to-3d",
                "input": {"image": f"data:image/png;base64,{image_b64}"},
            },
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]

        # Poll for completion
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            poll_resp = await client.get(
                f"{PIAPI_BASE_URL}/task/{task_id}",
                headers={"X-API-Key": api_key},
            )
            poll_resp.raise_for_status()
            data = poll_resp.json()

            if data["status"] == "completed":
                model_url = data["output"]["model_url"]
                break
            elif data["status"] == "failed":
                raise RuntimeError(f"Reconstruction failed: {data}")

            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(f"Reconstruction timed out after {timeout}s")

        # Download mesh
        dl_resp = await client.get(model_url)
        dl_resp.raise_for_status()

        safe_name = reference.replace("/", "_").replace(" ", "_")
        out_path = cache_dir / f"{safe_name}.glb"
        out_path.write_bytes(dl_resp.content)
        return out_path
