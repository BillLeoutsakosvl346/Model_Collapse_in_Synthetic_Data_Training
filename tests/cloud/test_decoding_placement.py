from model_collapse.cloud.execution import configured_placements


def test_gpu_and_data_center_pin_do_not_fall_back_to_stale_other_region():
    choices = [{"gpu": "NVIDIA RTX PRO 4500 Blackwell", "data_center": "EU-CZ-1"},
               {"gpu": "NVIDIA RTX PRO 4500 Blackwell", "data_center": "EU-RO-1"},
               {"gpu": "NVIDIA RTX PRO 4000 Blackwell", "data_center": "EU-RO-1"}]
    c = {"required_gpu": "NVIDIA RTX PRO 4500 Blackwell", "required_data_center": "EU-RO-1"}
    assert configured_placements(c, choices) == [choices[1]]
    assert configured_placements(c, choices[:1]) == []
    assert configured_placements({}, choices) == choices
    assert configured_placements({"required_gpu": c["required_gpu"]}, choices) == choices[:2]


def test_explicit_host_cuda_pin_preserves_legacy_floor():
    from model_collapse.cloud.execution import configured_gpu_request
    import pytest
    choice = {"gpu": "NVIDIA RTX PRO 4500 Blackwell"}
    assert configured_gpu_request({}, choice) == {"id": choice["gpu"], "count": 1, "minCudaVersion": "12.8"}
    assert configured_gpu_request({"required_host_cuda_version": "13.2"}, choice) == {
        "id": choice["gpu"], "count": 1, "allowedCudaVersions": ["13.2"]}
    for invalid in (13.2, "13", True):
        with pytest.raises(ValueError):
            configured_gpu_request({"required_host_cuda_version": invalid}, choice)


def test_auth_handshake_timeout_retries_but_rejected_keys_do_not(monkeypatch):
    from model_collapse.cloud import execution
    import paramiko
    import pytest
    clock, calls = [100.], []
    monkeypatch.setattr(execution, "now", lambda: clock[0])
    monkeypatch.setattr(execution.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    def timeout_then_ok():
        calls.append(True)
        if len(calls) < 3:
            raise paramiko.AuthenticationException("Authentication timeout.")
        return "progress"
    assert execution._transport_read(timeout_then_ok, 130) == "progress"
    assert len(calls) == 3
    calls.clear()
    def rejected():
        calls.append(True)
        raise paramiko.AuthenticationException("Authentication failed.")
    with pytest.raises(paramiko.AuthenticationException):
        execution._transport_read(rejected, 130)
    assert len(calls) == 1


def test_explicit_container_disk_price_preserves_legacy_default():
    from model_collapse.cloud.execution import configured_container_disk,CONTAINER_HOURLY
    import pytest
    assert configured_container_disk({})==(20,CONTAINER_HOURLY)
    assert configured_container_disk({'container_disk_gb':30})==(30,30*.10/730)
    for size in (True,19,101,30.):
        with pytest.raises(ValueError):configured_container_disk({'container_disk_gb':size})
