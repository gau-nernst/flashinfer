# Copyright (c) 2024 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import patch

import pytest

from flashinfer.comm import mnnvl


@pytest.mark.parametrize(
    "fabric_uuid,state,supported,expected",
    [
        ("00112233445566778899aabbccddeeff", 3, 1, True),
        ("112233445566778899aabbccddeeff00", 3, 1, True),
        ("00000000000000000000000000000001", 3, 1, True),
        ("00000000000000000000000000000000", 3, 1, False),
        ("00112233445566778899aabbccddeeff", 2, 1, False),
        ("00112233445566778899aabbccddeeff", 3, 0, False),
    ],
)
def test_is_mnnvl_fabric_supported(fabric_uuid, state, supported, expected):
    fabric_info = mnnvl.pynvml.c_nvmlGpuFabricInfoV_t()
    fabric_info.state = state
    fabric_info.clusterUuid[:] = bytes.fromhex(fabric_uuid)
    with (
        patch.object(
            mnnvl.cuda,
            "cuDeviceGetAttribute",
            return_value=(mnnvl.cuda.CUresult.CUDA_SUCCESS, supported),
        ),
        patch.object(mnnvl.pynvml, "nvmlInit") as init,
        patch.object(mnnvl.pynvml, "nvmlShutdown") as shutdown,
        patch.object(mnnvl.pynvml, "nvmlDeviceGetHandleByIndex"),
        patch.object(mnnvl.pynvml, "c_nvmlGpuFabricInfoV_t", return_value=fabric_info),
        patch.object(mnnvl.pynvml, "nvmlDeviceGetGpuFabricInfoV"),
    ):
        assert mnnvl.is_mnnvl_fabric_supported(0) is expected
        if supported:
            init.assert_called_once()
            shutdown.assert_called_once()
        else:
            init.assert_not_called()
            shutdown.assert_not_called()


@pytest.mark.parametrize("local_error,peer_ready", [(False, 1), (True, 1), (False, 0)])
def test_initialization_errors_are_agreed_before_next_collective(
    local_error, peer_ready
):
    from unittest.mock import Mock

    comm = Mock(allgather=lambda ready: [ready, peer_ready])
    if local_error or not peer_ready:
        with (
            pytest.raises(RuntimeError, match="initialization failed"),
            mnnvl._collective_init_stage(comm),
        ):
            if local_error:
                raise ValueError("local allocation failed")
    else:
        with mnnvl._collective_init_stage(comm):
            pass


@pytest.mark.parametrize("local_fabric,peer_fabric", [(True, False), (False, True)])
def test_handle_exchange_rejects_inconsistent_fabric_support(local_fabric, peer_fabric):
    from unittest.mock import Mock

    comm = Mock(
        allgather=Mock(side_effect=[[1, 1], [int(local_fabric), int(peer_fabric)]])
    )
    with (
        patch.object(mnnvl, "is_mnnvl_fabric_supported", return_value=local_fabric),
        pytest.raises(RuntimeError, match="disagree"),
    ):
        mnnvl.make_handle_exchanger(comm, 0, 2, 0)


def test_posix_handles_reject_cross_node_groups_before_exchange():
    from unittest.mock import Mock

    comm = Mock(allgather=Mock(side_effect=[[1, 1], [0, 0], [b"node0", b"node1"]]))
    with (
        patch.object(mnnvl, "is_mnnvl_fabric_supported", return_value=False),
        pytest.raises(RuntimeError, match="node-local"),
    ):
        mnnvl.make_handle_exchanger(comm, 0, 2, 0)
