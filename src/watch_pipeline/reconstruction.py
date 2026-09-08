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
    timeout: float = 300.0,
) -> Path:
    cached = get_cached_mesh(reference, cache_dir)
    if cached is not None:
        print(f"  Using cached mesh: {cached}")
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    image_b64 = base64.b64encode(Path(photo_path).read_bytes()).decode()
    suffix = Path(photo_path).suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    headers = {"x-api-key": api_key}

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Submit image-to-3D task via PiAPI TRELLIS
        print("  Submitting image-to-3D task to PiAPI TRELLIS...")
        resp = await client.post(
            f"{PIAPI_BASE_URL}/task",
            headers=headers,
            json={
                "model": "Qubico/trellis",
                "task_type": "image-to-3d",
                "input": {
                    "images": [f"data:{mime};base64,{image_b64}"],
                    "ss_sampling_steps": 12,
                    "slat_sampling_steps": 12,
                    "ss_guidance_strength": 7.5,
                    "slat_guidance_strength": 3,
                    "seed": 0,
                },
            },
        )
        resp.raise_for_status()
        resp_data = resp.json()
        print(f"  Response: {resp_data}")
        # PiAPI may nest under "data"
        task_id = resp_data.get("task_id") or resp_data.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {resp_data}")
        print(f"  Task created: {task_id}")

        # Poll for completion
        start = time.monotonic()
        last_status = ""
        while time.monotonic() - start < timeout:
            poll_resp = await client.get(
                f"{PIAPI_BASE_URL}/task/{task_id}",
                headers=headers,
            )
            poll_resp.raise_for_status()
            data = poll_resp.json().get("data", poll_resp.json())
            status = data.get("status", "unknown")

            if status != last_status:
                print(f"  Status: {status}")
                last_status = status

            if status == "completed":
                model_url = data["output"]["model_file"]
                break
            elif status == "failed":
                raise RuntimeError(f"Reconstruction failed: {data}")

            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(f"Reconstruction timed out after {timeout}s")

        # Download GLB mesh
        print("  Downloading GLB mesh...")
        dl_resp = await client.get(model_url)
        dl_resp.raise_for_status()

        safe_name = reference.replace("/", "_").replace(" ", "_")
        out_path = cache_dir / f"{safe_name}.glb"
        out_path.write_bytes(dl_resp.content)
        print(f"  Saved mesh: {out_path} ({len(dl_resp.content)} bytes)")
        return out_path
