import numpy as np

from mujoco_py import MjViewer
from mj_envs.envs import env_base
import os
import collections

VIZ = False

class KitchenBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS_AND_WEIGHTS = {
        "hand_jnt": 1.0,
        "objs_jnt": 1.0,
        "goal": 1.0,
        "goal_err": 1.0,
        "approach_err": 1.0,
    }
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "goal": 1.0,
        "bonus": 0.0, #0.5,
        "pose": 0, #0*0.01,
        "approach": 0.1 #0.5,
    }

    def __init__(self, model_path, config_path,
                    robot_jnt_names, obj_jnt_names, goal, interact_site,
                    obs_keys_wt = DEFAULT_OBS_KEYS_AND_WEIGHTS,
                    rwd_keys_wt = DEFAULT_RWD_KEYS_AND_WEIGHTS,
                    **kwargs):

        if VIZ:
            from vtils.plotting.srv_dict import srv_dict
            self.dict_plot = srv_dict()

        # get sims
        self.sim = env_base.get_sim(model_path=model_path)
        self.sim_obsd = env_base.get_sim(model_path=model_path)

        # backward compatible TODO
        self.act_mid = np.zeros(self.sim.model.nu)
        self.act_amp = 2.0 * np.ones(self.sim.model.nu)

        # configure env-site
        self.grasp_sid = self.sim.model.site_name2id('end_effector')
        self.interact_sid = self.sim.model.site_name2id(interact_site)

        # configure env-robot
        self.robot_dofs = []
        self.robot_ranges = []
        for jnt_name in robot_jnt_names:
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.robot_dofs.append(self.sim.model.jnt_dofadr[jnt_id])
            self.robot_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.robot_dofs = np.array(self.robot_dofs)
        self.robot_meanpos = np.mean(self.robot_ranges)

        # configure env-objs
        self.obj_dofs = []
        self.obj_ranges = []
        
        #print("+++++++++++++++++++")
        for jnt_name in obj_jnt_names:
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.obj_dofs.append(self.sim.model.jnt_dofadr[jnt_id])
            self.obj_ranges.append(self.sim.model.jnt_range[jnt_id])    
            #print(jnt_name, self.sim.model.jnt_dofadr[jnt_id], self.sim.model.jnt_range[jnt_id])
        #print("+++++++++++++++++++")
        

        self.obj_dofs = np.array(self.obj_dofs)
        self.obj_ranges = np.array(self.obj_ranges)
        self.obj_ranges = self.obj_ranges[:,1] - self.obj_ranges[:,0]

        # configure env-goal
        self.set_goal(goal)
        self.viewer = None

        # get env
        env_base.MujocoEnv.__init__(self,
                                sim = self.sim,
                                sim_obsd = self.sim_obsd,
                                frame_skip = 40,
                                config_path = config_path,
                                obs_keys = list(obs_keys_wt.keys()),
                                rwd_keys_wt = rwd_keys_wt,
                                rwd_mode = "dense",
                                act_mode = "vel",
                                act_normalized = True,
                                is_hardware = False,
                                robot_name = "Franka_kitchen_sim",
                                obs_range = (-8, 8),
                                **kwargs)

        self.init_qpos = self.sim.model.key_qpos[0].copy()


    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict['t'] = np.array([sim.data.time])

        ### raw observations
        obs_dict['hand_jnt'] = sim.data.qpos[self.robot_dofs].copy()
        obs_dict['objs_jnt'] = sim.data.qpos[self.obj_dofs].copy()
        obs_dict['goal'] = self.goal.copy()
        #print(obs_dict['objs_jnt'],obs_dict['goal'])
        ### deltas
        obs_dict['goal_err'] = obs_dict['goal']-obs_dict['objs_jnt'] #??? Kettle has quaternions
        obs_dict['approach_err'] = self.sim.data.site_xpos[self.interact_sid] - self.sim.data.site_xpos[self.grasp_sid]
        obs_dict['pose_err'] = self.robot_meanpos-obs_dict['hand_jnt']

        return obs_dict


    def get_reward_dict(self, obs_dict):
        goal_dist = np.abs(obs_dict['goal_err'])

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('goal',    -np.sum(goal_dist, axis=-1)),
            ('bonus',   np.sum(goal_dist < 0.75*self.obj_ranges, axis=-1) + np.sum(goal_dist < 0.25*self.obj_ranges, axis=-1)),
            ('pose',    -np.sum(np.abs(obs_dict['pose_err']), axis=-1)),
            ('approach',-np.linalg.norm(obs_dict['approach_err'], axis=-1)),
            # Must keys
            ('sparse',  -np.sum(goal_dist[0][0], axis=-1)),
            ('solved',  np.all(goal_dist < 0.1*self.obj_ranges)),
            ('done',    False),
        ))
        #print(goal_dist[0][0], 0.1*self.obj_ranges, np.all(goal_dist[0][0] < 0.1*self.obj_ranges))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        if self.mujoco_render_frames and VIZ:
            self.dict_plot.append(rwd_dict, self.rwd_keys_wt)
            # self.dict_plot.append(rwd_dict)

        return rwd_dict


    def set_goal(self, goal=None):
        if goal is not None:
            assert len(goal) == len(self.obj_dofs), "Check size of provided goal"
            self.goal = goal
        else: # treat current sim as goal
            self.goal = self.sim.data.qpos[self.obj_dofs].copy()

    def render(self, mode='human'):
        ''' Render the environment to the screen '''
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            # Turn all the geom groups on
            self.viewer.vopt.geomgroup[:] = 1
            # Set camera if specified
            if mode == 'human':
                self.viewer.cam.fixedcamid = -1
                #self.viewer.cam.type = const.CAMERA_FREE
            else:
                self.viewer.cam.fixedcamid = self.model.camera_name2id(mode)
                #self.viewer.cam.type = const.CAMERA_FIXED
        
        self.viewer.render() 

class KitchenFetchFixed(KitchenBase):

    def __init__(self,
                goal=None,
                interact_site="end_effector"):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        KitchenBase.__init__(self,
            model_path = curr_dir + '/assets/franka_kitchen.xml',
            config_path = curr_dir + '/assets/franka_kitchen.config',
            robot_jnt_names = ('panda0_joint1', 'panda0_joint2', 'panda0_joint3', 'panda0_joint4', 'panda0_joint5', 'panda0_joint6', 'panda0_joint7', 'panda0_finger_joint1', 'panda0_finger_joint2'),
            obj_jnt_names = ('knob_Joint_1', 'knob_Joint_2', 'knob_Joint_3', 'knob_Joint_4', 'lightswitch_joint', 'slidedoor_joint', 'leftdoorhinge', 'rightdoorhinge', 'microjoint'),
            goal=goal,
            interact_site=interact_site)
        
        #properties = dir(self.sim.model.get_mjb().find(self.sim.model.body_name2id('microwave'))) #"./body[@name='microwave']"))
        #for p in dir(self.sim.model):
            #if 'body' in p: print(p)
            #if 'get' in p: print(p)
        #print(self.sim.model.body_name2id('microwave'))

        #print(self.sim.model.body_ipos[self.sim.model.body_name2id('microwave')])
        #print(self.sim.model.body_iquat[self.sim.model.body_name2id('microwave')])

        #self.sim.model.body_ipos.flags.writeable = True
        #self.sim.model.body_ipos[self.sim.model.body_name2id('microwave')] = [-0.65,  -0.025,  1.6 ]
        #self.sim.model.body_iquat[self.sim.model.body_name2id('microwave')] = 
        #print(self.sim.model.body_ipos[self.sim.model.body_name2id('microwave')])
        
