"""Small REST v2 client. Never put API keys in URLs or error output."""
from __future__ import annotations

import json
import http.client
import os
import re
import ssl
import time
import urllib.error
import urllib.request

from ..common import load_secrets


class Runpod:
    def __init__(self, key=None):
        load_secrets()
        self.key = key or os.environ.get("RUNPOD_API_KEY")
        if not self.key:
            raise RuntimeError("RUNPOD_API_KEY is missing from the ignored .env")

    def request(self, method, path, body=None, *, graphql=False):
        url = "https://api.runpod.io" + ("/graphql" if graphql else "/v2" + path)
        req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        query = body.get("query") if isinstance(body, dict) else None
        read_query = (graphql and method.upper() == "POST" and isinstance(query, str)
                      and re.match(r"\s*query\b", query, re.IGNORECASE)
                      and not re.search(r"\b(?:mutation|subscription)\b", query, re.IGNORECASE))
        read_only = method.upper() == "GET" or bool(read_query)
        attempts = 4 if read_only else 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                if read_only and (e.code == 429 or 500 <= e.code < 600) and attempt + 1 < attempts:
                    e.close()
                    time.sleep(min(2 ** attempt, 4))
                    continue
                try:
                    raw = e.read(2000).decode(errors="replace")
                finally:
                    e.close()
                for secret in (self.key, os.environ.get("HF_TOKEN")):
                    if secret:
                        raw = raw.replace(secret, "[REDACTED]")
                raise RuntimeError(f"Runpod {method} {path}: HTTP {e.code}: {raw[:1000]}") from None
            except (OSError, EOFError, http.client.IncompleteRead) as e:
                certificate_error = isinstance(e, ssl.SSLCertVerificationError) or isinstance(
                    getattr(e, "reason", None), ssl.SSLCertVerificationError)
                if certificate_error or attempt + 1 == attempts:
                    raise
                time.sleep(min(2 ** attempt, 4))

    def account(self):
        d = self.request("POST", "", {"query": "query { myself { clientBalance currentSpendPerHr isAutoPayEnabled enableBackupPayments } }"}, graphql=True)
        if d.get("errors"):
            raise RuntimeError("Runpod account billing read failed")
        return d["data"]["myself"]

    def pod(self, pod_id):
        return self.request("GET", "/pods/" + pod_id)

    def billing(self):
        return self.request("GET", "/billing?lastN=1&bucketSize=day")

    def stop(self, pod_id):
        for attempt in range(8):
            try:
                p = self.pod(pod_id)
                if p["status"] in ("EXITED", "TERMINATED"):
                    return {"id": pod_id, "status": p["status"], "cost": p["cost"]}
                if "stop" in p.get("actions", []):
                    self.request("POST", f"/pods/{pod_id}/action", {"action": "stop"})
            except (RuntimeError, OSError):
                pass
            time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"Could not verify stopped compute for Pod {pod_id}")

    def candidates(self, *, network_storage=True):
        data = self.request("GET", "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8")
        dcs = {d["id"]: d for d in self.request("GET", "/catalog/datacenters")["dataCenters"]}
        choices = []
        for g in data["gpus"]:
            if g["manufacturer"] != "NVIDIA" or g["memory"] < 16 or not g["secure"] or g.get("availability") == "NONE":
                continue
            for dc in g.get("dataCenters", []):
                if dc["availability"] != "NONE" and (not network_storage or "STANDARD" in dcs[dc["id"]]["networkVolumeTypes"]):
                    choices.append({"gpu": g["id"], "vram_gb": g["memory"], "hourly_usd": g["price"]["secure"],
                                    "data_center": dc["id"], "availability": dc["availability"]})
        return sorted(choices, key=lambda x: (x["hourly_usd"], x["data_center"]))
