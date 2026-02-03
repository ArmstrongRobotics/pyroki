import numpy as np
import torch
import jaxlie
import pyroki as pk
import pyroki.pyroki_snippets as pks
from scipy.spatial.transform import Rotation as R
# Pre-import these to register the robot description
import robot_descriptions
import robot_descriptions.yam_description
from robot_descriptions.loaders.yourdfpy import load_robot_description


class I2RTPyroki():
    def __init__(self):
        urdf = load_robot_description("yam_description")
        self.pyro = pk.Robot.from_urdf(urdf)
        self.ee_name = "link_6"
        self.ee_idx = self.pyro.links.names.index(self.ee_name)

        # Warmstart pyroik jit
        self.ik(np.eye(4))
        self.fk(np.random.random(len(self.pyro.joints.names)))

    def ik(self, ee_pose_matrix):
        assert ee_pose_matrix.shape == (4, 4), "EE pose matrix must be 4x4, got shape: {}".format(ee_pose_matrix.shape)
        return pks.solve_ik(
            robot=self.pyro,
            target_link_name=self.ee_name,
            target_position=ee_pose_matrix[:3, 3],
            target_wxyz=R.from_matrix(ee_pose_matrix[:3, :3]).as_quat(scalar_first=True),
        )[::-1]  # Return reverse joint order since pyroki uses reversed joint order

    def fk(self, joints, as_matrix=True):
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        # Add batch dimension if not present
        added_batch_dim = False
        if len(joints.shape) == 1:
            joints = joints[None, :]
            added_batch_dim = True
        assert joints.shape[-1] == len(self.pyro.joints.names), "Joint dimension mismatch, input joints shape: {}, expected shape: {}".format(joints.shape, len(self.pyro.joints.names))
        ee_matrix = np.array(jaxlie.SE3(self.pyro.forward_kinematics(joints[..., ::-1])[:, -1]).as_matrix())  # Reverse joint order since pyroki uses reversed joint order
        if as_matrix:
            ret = ee_matrix
        else:
            # Return as xyz + rot6d
            trans = ee_matrix[..., :3, 3]
            rotx = ee_matrix[..., :3, 0]
            roty = ee_matrix[..., :3, 1]
            ret = np.concatenate([trans, rotx, roty], axis=-1)
        if added_batch_dim:
            ret = ret[0]
        return ret