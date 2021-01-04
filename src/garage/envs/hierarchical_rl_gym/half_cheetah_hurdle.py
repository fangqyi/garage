import os

import numpy as np
from gym import utils
from garage.envs.hierarchical_rl_gym import mujoco_env
import mujoco_py
import cv2

PROJECT_PATH = os.path.dirname(os.path.realpath(os.path.join(__file__, '..')))
MODELS_PATH = os.path.abspath(os.path.join(PROJECT_PATH, '/assets'))

class HalfCheetahEnv_Hurdle(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.interval_time = 0
        mujoco_env.MujocoEnv.__init__(self, MODELS_PATH+'half_cheetah_hurdle.xml', 5)
        utils.EzPickle.__init__(self)
        self.ob_type = ['joint']
        self.ob_shape = {'joint': [self.observation_space.shape[0]]}
        self._viewers = {}



    def step(self, action):
        xposbefore = self.sim.data.qpos[0] # self.sim.data.qpos.flat[0]
        # print(xposbefore)
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0] # self.sim.data.qpos.flat[0], the x axis of the halfcheetah
        zposafter = self.sim.data.qpos[1]
        yposafter = self.sim.data.qpos[2]
        # print(xposafter)
        ob = self._get_obs()
        # print(ob)
        reward_ctrl = - 0.1 * np.square(action).sum() # feet not toward sky
        reward_run = (xposafter - xposbefore)/self.dt # walk forward
        reward_success = -1
        reward_failed = 0

        # print(xposafter, zposafter, yposafter)
        x_hurdle = 1.5 # 1 #
        valid_distance = 1.7 # 1.2 #
        invalid_distance = 6

        # if xposafter < x_hurdle - invalid_distance or yposafter > 3:# or yposafter < -2.85:
        #     reward_failed = -100
        #     reward = reward_ctrl + reward_run + reward_success + reward_failed
        #     print('failed')
        #     return ob, reward, True, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
        #                                   success=0, reward_failed=reward_failed)
        # if zposafter < -0.45 and yposafter > 2.5:
        #     reward_failed = -1

        # if xposafter > x_hurdle + valid_distance:
        #     reward_success = 2

        if xposafter > x_hurdle + valid_distance and yposafter < 3 and yposafter > -2.85: # the agent must stand
            reward_success = 500
            reward = reward_ctrl + reward_run + reward_success + reward_failed
            print('Success')
            return ob, reward, True, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                           success=1, reward_failed=reward_failed)

        reward = reward_ctrl + reward_run + reward_success + reward_failed
        return ob, reward, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                    success=0, reward_failed=reward_failed)

    def _get_obs(self):

        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        # if self._exclude_current_positions_from_observation:
        #     position = position[1:]

        # observation = np.concatenate((position, velocity)).ravel()
        observation = position.ravel()
        return observation

        # # curb_obs = self._get_curb_observation()
        # return np.concatenate([
        #     # self.sim.data.qpos.flat[1:], # 8-dimensional
        #     self.sim.data.qpos.flat, # 9-dimensional, modified by HEGSNS
        #     # self.sim.data.qvel.flat, # 9-dimensional
        #     # curb_obs,
        #     # self.sim.data.qpos
        # ])

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :],
            }
        return {
            'joint': ob[:],
        }


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        ## added by HEGSNS
        self.interval_time = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.55

    def render_frame(self): # , camera_name=None):
        viewer = self._get_viewer() # mode="human") # , camera_name=camera_name)
        self.viewer_setup()
        img = viewer._read_pixels_as_in_window()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    # TODO LJX add camera_name=camera_name
    def _get_viewer(self): # , mode): # , camera_name=None):
        mode = "human"
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim) #, camera_name=camera_name)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    ## added by HEGSNS
    # def _put_curbs(self):
    #     if self._curbs is None:
    #         geom_name_list = self.model.geom_names
    #         self._curbs = {'pos': [], 'size': []}
    #         self._num_curbs = len(np.where(['curb' in name for name in geom_name_list])[0])
    #         for i in range(self._num_curbs): # curb1-curb5
    #             idx = self.model.geom_name2id('curb{}'.format(i+1))
    #             r = self._config["curb_randomness"]
    #             self.model.geom_pos[idx][0] = self._curbs_x[i] + np.random.uniform(low=-r, high=r)
    #             self.model.geom_pos[idx][2] = self._config["curb_height"]
    #             self.model.geom_size[idx][2] = self._config["curb_height"]
    #             self._curbs['pos'].append(self.model.geom_pos[idx][0])
    #             self._curbs['size'].append(self.model.geom_size[idx][0])
    #         self._curbs['pos'].append(1000)
    #         self._curbs['size'].append(1)

    # def _get_curb_observation(self):
    #     if self._curbs is None:
    #         self._put_curbs()
    #     # x_agent = self.sim.data.qpos[0] # self._get_walker2d_pos()
    #     x_agent = self.data.qpos[0]
    #     if x_agent > self._curbs['pos'][self._stage] + self._curbs['size'][self._stage] + \
    #             self._config["success_dist_after_curb"]:
    #         self._stage += 1
    #     if self._stage >= self._num_curbs:
    #         return (5.1, 5.2)
    #     else:
    #         curb_start = self._curbs['pos'][self._stage] - self._curbs['size'][self._stage]
    #         curb_end = self._curbs['pos'][self._stage] + self._curbs['size'][self._stage]
    #         if curb_start - x_agent > self._config['curb_detect_dist']:
    #             return (5.1, 5.2)
    #         return (curb_start - x_agent, curb_end - x_agent)
