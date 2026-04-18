import asyncio
import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import aiohttp

from ..utils.audio import validate_audio_file


logger = logging.getLogger(__name__)


class MimoTTS:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        *,
        fmt: str = "wav",
        voice_id: str = "mimo_default",
        style: str = "",
        sample_rate: int = 24000,
        max_retries: int = 2,
        timeout: int = 30,
    ):
        self.api_url = api_url.strip().rstrip("/") or "https://api.xiaomimimo.com/v1"
        self.api_key = api_key.strip()
        self.model = (model or "mimo-v2-tts").strip()
        self.format = (fmt or "wav").strip().lower() or "wav"
        self.voice_id = (voice_id or "mimo_default").strip() or "mimo_default"
        self.style = (style or "").strip()
        self.sample_rate = max(1, int(sample_rate or 24000))
        self.max_retries = max(0, int(max_retries))
        self.timeout = max(5, int(timeout))
        self._session: Optional[aiohttp.ClientSession] = None

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=client_timeout)

    @staticmethod
    async def _write_bytes(path: Path, content: bytes) -> None:
        def _write():
            with open(path, "wb") as f:
                f.write(content)

        await asyncio.to_thread(_write)

    @staticmethod
    def _emotion_to_style(emotion: Optional[str]) -> str:
        return {
            "happy": "开心",
            "sad": "悲伤",
            "angry": "生气",
            "neutral": "平静",
        }.get(str(emotion or "").strip().lower(), "")

    def _build_styled_text(self, text: str, emotion: Optional[str] = None) -> str:
        content = (text or "").strip()
        if not content:
            return ""

        styles = []
        if self.style:
            styles.append(self.style)
        emotion_style = self._emotion_to_style(emotion)
        if emotion_style and emotion_style not in styles:
            styles.append(emotion_style)

        if styles:
            return f"<style>{' '.join(styles)}</style>{content}"
        return content

    def _build_payload(self, text: str, *, voice: str, audio_format: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "assistant",
                    "content": text,
                }
            ],
            "modalities": ["text", "audio"],
            "audio": {
                "format": audio_format,
                "voice": voice,
            },
            "stream": False,
        }

    async def synth(
        self,
        text: str,
        voice: str,
        out_dir: Path,
        speed: Optional[float] = None,
        *,
        emotion: Optional[str] = None,
    ) -> Optional[Path]:
        _ = speed
        _ = self.sample_rate
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.error("MimoTTS: missing api key")
            return None

        effective_voice = (voice or self.voice_id or "mimo_default").strip() or "mimo_default"
        styled_text = self._build_styled_text(text, emotion=emotion)
        if not styled_text:
            logger.error("MimoTTS: empty assistant content")
            return None

        request_audio_format = self.format if self.format in {"wav", "mp3"} else "wav"

        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "text": styled_text,
                    "voice": effective_voice,
                    "model": self.model,
                    "format": request_audio_format,
                    "style": self.style,
                    "emotion": emotion,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]

        out_path = out_dir / f"{cache_key}.{request_audio_format}"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        payload = self._build_payload(styled_text, voice=effective_voice, audio_format=request_audio_format)
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        request_url = f"{self.api_url}/chat/completions"

        await self._ensure_session()
        last_error = None
        backoff = 1.0

        for attempt in range(1, self.max_retries + 2):
            try:
                assert self._session is not None
                async with self._session.post(request_url, headers=headers, json=payload) as resp:
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        data = await resp.text()

                    if not (200 <= resp.status < 300):
                        last_error = f"http {resp.status}: {data}"
                        if resp.status in (429,) or 500 <= resp.status < 600:
                            if attempt <= self.max_retries:
                                await asyncio.sleep(backoff)
                                backoff = min(backoff * 2, 8)
                                continue
                        break

                    if not isinstance(data, dict):
                        last_error = f"unexpected response: {data}"
                        break

                    choices = data.get("choices") or []
                    if not choices:
                        last_error = "empty choices"
                        break

                    message = (choices[0] or {}).get("message") or {}
                    audio = message.get("audio") or {}
                    audio_base64 = str(audio.get("data") or "").strip()
                    if not audio_base64:
                        last_error = f"missing audio data: {data}"
                        break

                    raw = base64.b64decode(audio_base64)
                    await self._write_bytes(out_path, raw)

                    if not await validate_audio_file(out_path, expected_format=request_audio_format):
                        last_error = "audio file validation failed"
                        break
                    return out_path
            except Exception as e:
                last_error = str(e)
                if attempt <= self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    continue
                break

        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except Exception:
            pass

        logger.error("MimoTTS synth failed: %s", last_error)
        return None
