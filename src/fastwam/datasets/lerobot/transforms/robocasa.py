from typing import Dict, List

import torch


class RobocasaSliceTransform:
    """Drop constant all-zero dimensions from robocasa365 (PandaOmron) action/state.

    Source layout (before transform):
      state 16D:
        [0:3]   base position (x, y, z)
        [3:7]   base rotation quaternion (0, 0, sin(yaw), cos(yaw))  -> [3:5] are constant 0
        [7:10]  end-effector position (relative)
        [10:14] end-effector rotation quaternion (relative)
        [14:16] gripper qpos
      action 12D:
        [0:4]   base_motion (Vx, Vy, yaw, 0)  -> [3] is constant 0
        [4:5]   control_mode (discrete, +1 base / -1 eef)
        [5:8]   end-effector position delta
        [8:11]  end-effector rotation delta
        [11:12] gripper_close (discrete, +1 / -1)

    After transform:
      state 14D: drop indices [3, 4]
      action 11D: drop index [3]

    forward:  slice raw dims -> reduced dims (used in preprocess & stats computation)
    backward: scatter reduced dims -> raw dims, filling dropped dims with 0 (used in postprocess)
    """

    STATE_RAW_DIM = 16
    ACTION_RAW_DIM = 12
    STATE_KEEP: List[int] = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 14 dims
    ACTION_KEEP: List[int] = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]  # 11 dims

    def __init__(self, action_key: str = "default", state_key: str = "default"):
        self.action_key = action_key
        self.state_key = state_key

    @staticmethod
    def _slice(x: torch.Tensor, keep: List[int]) -> torch.Tensor:
        dim = x.dim() - 1
        index = torch.as_tensor(keep, dtype=torch.long, device=x.device)
        return x.index_select(dim, index)

    @staticmethod
    def _scatter(x: torch.Tensor, keep: List[int], raw_dim: int) -> torch.Tensor:
        dim = x.dim() - 1
        out_shape = list(x.shape)
        out_shape[dim] = raw_dim
        full = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        index = torch.as_tensor(keep, dtype=torch.long, device=x.device)
        full.index_copy_(dim, index, x)
        return full

    def forward(self, batch: Dict) -> Dict:
        state = batch["state"][self.state_key]
        assert state.shape[-1] == self.STATE_RAW_DIM, (
            f"RobocasaSliceTransform expects state last dim {self.STATE_RAW_DIM}, got {state.shape[-1]}"
        )
        batch["state"][self.state_key] = self._slice(state, self.STATE_KEEP)

        # for close-loop eval, "action" may not be in batch
        if "action" in batch:
            action = batch["action"][self.action_key]
            assert action.shape[-1] == self.ACTION_RAW_DIM, (
                f"RobocasaSliceTransform expects action last dim {self.ACTION_RAW_DIM}, got {action.shape[-1]}"
            )
            batch["action"][self.action_key] = self._slice(action, self.ACTION_KEEP)

        return batch

    def backward(self, batch: Dict) -> Dict:
        state = batch["state"][self.state_key]
        assert state.shape[-1] == len(self.STATE_KEEP), (
            f"RobocasaSliceTransform backward expects state last dim {len(self.STATE_KEEP)}, got {state.shape[-1]}"
        )
        batch["state"][self.state_key] = self._scatter(state, self.STATE_KEEP, self.STATE_RAW_DIM)

        if "action" in batch:
            action = batch["action"][self.action_key]
            assert action.shape[-1] == len(self.ACTION_KEEP), (
                f"RobocasaSliceTransform backward expects action last dim {len(self.ACTION_KEEP)}, got {action.shape[-1]}"
            )
            batch["action"][self.action_key] = self._scatter(action, self.ACTION_KEEP, self.ACTION_RAW_DIM)

        return batch
