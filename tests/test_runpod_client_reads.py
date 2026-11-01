import http.client
import io
import json
import ssl
import urllib.error

import pytest

from model_collapse.runpod_api import Runpod
from model_collapse import runpod_api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(runpod_api, "load_secrets", lambda: None)
    monkeypatch.setattr(runpod_api.time, "sleep", lambda _: None)
    return Runpod(key="test-not-a-real-key")


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def test_transient_ssl_eof_get_is_retried_then_succeeds(client, monkeypatch):
    calls = []
    def request(req, timeout):
        calls.append(req.get_method())
        if len(calls) < 3:
            raise urllib.error.URLError(ssl.SSLEOFError("temporary unexpected EOF"))
        return response({"pods": []})
    monkeypatch.setattr(runpod_api.urllib.request, "urlopen", request)
    assert client.request("GET", "/pods") == {"pods": []}
    assert calls == ["GET"] * 3


def test_account_query_is_a_read_and_can_retry(client, monkeypatch):
    calls = []
    def request(req, timeout):
        calls.append(json.loads(req.data)["query"])
        if len(calls) == 1: raise TimeoutError("temporary timeout")
        return response({"data": {"myself": {"clientBalance": 10}}})
    monkeypatch.setattr(runpod_api.urllib.request, "urlopen", request)
    assert client.account() == {"clientBalance": 10}
    assert len(calls) == 2 and all(q.startswith("query") for q in calls)


@pytest.mark.parametrize("method,path,body,graphql", [
    ("POST", "/pods", {"name": "new"}, False),
    ("POST", "/network-volumes", {"name": "new"}, False),
    ("PUT", "/account/ssh-keys", {"keys": []}, False),
    ("DELETE", "/pods/existing", None, False),
    ("POST", "", {"query": "mutation { stopPod(input:{podId:\"pod\"}) { id } }"}, True),
    ("POST", "", {"query": "query Read { myself { clientBalance } } mutation Write { stopPod(input:{podId:\"pod\"}) { id } }", "operationName": "Write"}, True),
    ("POST", "", {"query": "subscription { updates { id } }"}, True),
    ("POST", "", {"query": "{ myself { clientBalance } }"}, True),
])
def test_mutations_and_ambiguous_graphql_are_never_retried(client, monkeypatch, method, path, body, graphql):
    calls = []
    def request(req, timeout):
        calls.append(req.get_method())
        raise urllib.error.URLError("uncertain network outcome")
    monkeypatch.setattr(runpod_api.urllib.request, "urlopen", request)
    with pytest.raises(urllib.error.URLError): client.request(method, path, body, graphql=graphql)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [401, 403, 400])
def test_auth_denial_and_bad_requests_fail_without_retry_and_redact(client, monkeypatch, status):
    calls = []
    def request(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, status, "denied", {}, io.BytesIO(client.key.encode()))
    monkeypatch.setattr(runpod_api.urllib.request, "urlopen", request)
    with pytest.raises(RuntimeError) as error: client.request("GET", "/pods")
    assert len(calls) == 1
    assert client.key not in str(error.value) and "[REDACTED]" in str(error.value)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_http_read_errors_retry(client, monkeypatch, status):
    calls = []
    def request(req, timeout):
        calls.append(1)
        if len(calls) < 4:
            raise urllib.error.HTTPError(req.full_url, status, "temporary", {}, io.BytesIO(b"temporary"))
        return response({"id": "pod"})
    monkeypatch.setattr(runpod_api.urllib.request, "urlopen", request)
    assert client.pod("pod") == {"id": "pod"}
    assert len(calls) == 4


@pytest.mark.parametrize("failure", [EOFError("EOF"), TimeoutError("timeout"),
    urllib.error.URLError("disconnected"), http.client.IncompleteRead(b"partial", 10)])
def test_permanent_transport_read_is_bounded_to_four_attempts(client, monkeypatch, failure):
    calls = []
    def request(req, timeout):
        calls.append(1)
        raise failure
    monkeypatch.setattr(runpod_api.urllib.request, "urlopen", request)
    with pytest.raises(type(failure)): client.request("GET", "/pods")
    assert len(calls) == 4


def test_tls_certificate_validation_failure_is_not_retried(client, monkeypatch):
    calls = []
    def request(req, timeout):
        calls.append(1)
        raise urllib.error.URLError(ssl.SSLCertVerificationError("invalid certificate"))
    monkeypatch.setattr(runpod_api.urllib.request, "urlopen", request)
    with pytest.raises(urllib.error.URLError): client.request("GET", "/pods")
    assert len(calls) == 1


def test_http_server_error_does_not_retry_uncertain_pod_creation(client, monkeypatch):
    calls = []
    def request(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 503, "uncertain", {}, io.BytesIO(b"uncertain"))
    monkeypatch.setattr(runpod_api.urllib.request, "urlopen", request)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        client.request("POST", "/pods", {"name": "new"})
    assert len(calls) == 1
