"""以独立的图文请求调用视觉语言模型，返回结构化决策。"""

import base64
import json
from time import perf_counter
from urllib.parse import urlsplit

import cv2
import numpy as np
import requests


class VisionAPI:
    def __init__(self, endpoint, model, api_key=None, max_calls=20, max_tokens=384, *, thinking=None):
        url = urlsplit(endpoint)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("端点应是不含凭据和查询参数的 HTTP(S) 地址")
        if not model or max_calls < 1 or max_tokens < 1:
            raise ValueError("需要模型名和正的调用预算")
        endpoint = endpoint.rstrip("/")
        if not url.path.strip("/"):
            endpoint += "/v1"
        self.endpoint = endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"
        self.model = model
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.calls = 0
        self.last_usage = {}
        self.last_call = {}
        self.session = requests.Session()
        if url.hostname in {"127.0.0.1", "localhost", "::1"}:
            self.session.trust_env = False
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def ask(self, system: str, prompt: str, images: list[np.ndarray]) -> dict:
        if self.calls >= self.max_calls:
            raise RuntimeError("超过设定的模型调用预算")
        items = [{"type": "text", "text": prompt}]
        image_bytes = 0
        for image in images:
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("模型图像须为 H×W×3 的 uint8 RGB")
            ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError("图像编码失败")
            image_bytes += data.nbytes
            encoded = base64.b64encode(data).decode("ascii")
            items.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})

        payload = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": items}]}
        if self.thinking is not None:
            payload["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        self.last_usage = {}
        self.calls += 1
        started = perf_counter()
        self.last_call = {"request": self.calls, "images": len(images), "image_bytes": image_bytes}
        try:
            response = self.session.post(self.endpoint, json=payload, timeout=(10, 120), allow_redirects=False)
            self.last_call["http_status"] = response.status_code
            if response.status_code != 200:
                raise RuntimeError(f"模型接口返回 HTTP {response.status_code}")
            body = response.json()
            self.last_usage = {key: value for key, value in (body.get("usage") or {}).items()
                               if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                               and type(value) is int and value >= 0}
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("模型输出超过长度预算")
            decision = json.loads(choice["message"]["content"])
            if not isinstance(decision, dict):
                raise ValueError("模型未返回 JSON 对象")
            return decision
        except requests.RequestException as error:
            raise RuntimeError(f"模型请求失败：{type(error).__name__}") from None
        except (KeyError, TypeError, IndexError, json.JSONDecodeError):
            raise ValueError("模型接口响应缺少有效的 JSON 决策") from None
        finally:
            self.last_call["elapsed_seconds"] = round(perf_counter() - started, 4)

    def close(self):
        self.session.close()
