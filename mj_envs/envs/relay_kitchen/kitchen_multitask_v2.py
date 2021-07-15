import numpy as np

from mujoco_py import MjViewer
from mj_envs.envs import env_base
import os
import collections


import torch
from PIL import Image
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2

import clip

VIZ = False
device = "cuda" if torch.cuda.is_available() else "cpu"
DEMO_MODE = 'microwave' #'cabinet' #
#BASE_DIR = '/Users/yuchencui/Projects/active_learning/'
BASE_DIR = '/private/home/yuchencui/projects/active_learning/'

class DeepFeatureSimilarityRewardFunction(object):
    def __init__(self, mode ='clip') -> None:
        super().__init__()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if mode == 'clip': self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        else: self.model = models.resnet152(pretrained=True)
        self.mode = mode

        self.model = self.model.cuda()
        modules=list(self.model.children())[:-1]
        self.modelmodel=torch.nn.Sequential(*modules)
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.model.eval()

        data_dir = BASE_DIR + 'online_imgs/kitchen_micro_open-v3/'

        goal_img_paths = [os.path.join(data_dir+'/goal', f) for f in os.listdir(data_dir+'/goal') if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')] 
        init_img_paths = [os.path.join(data_dir+'/init', f) for f in os.listdir(data_dir+'/init') if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')] 
        #goal_img_paths = ['online_imgs/'+DEMO_MODE+'_open.jpg'] 
        #init_img_paths = ['online_imgs/'+DEMO_MODE+'_closed.jpg'] 

        goal_imgs = torch.stack([self.preprocess(Image.open(f)).to(device) for f in goal_img_paths])
        init_imgs = torch.stack([self.preprocess(Image.open(f)).to(device) for f in init_img_paths])

        #base_image = Image.open(BASE_DIR+"imgs/kitchen_micro_open-v2/kitchen_micro_open-v2_0_0.png")
        #base_image = Image.open("/Users/yuchencui/Projects/active_learning/franka_baselines/human_demo_imgs/friday_microwave_bottomknob_hinge_slide_kitchen_playdata_2019_06_28_12_12_01_demo_render/0000.png")
        base_image = Image.open(BASE_DIR+"kitchen_base_image.png")
        base_image = self.preprocess(self.crop_img(base_image)).unsqueeze(0).to(device)
        if self.mode == 'clip':
            goal_img_features = self.model.encode_image(goal_imgs)
            init_img_features = self.model.encode_image(init_imgs) 
            self.base_features = self.model.encode_image(base_image)
        else:
            goal_img_features = self.model(goal_imgs)
            init_img_features = self.model(init_imgs)
            self.base_features = self.model(base_image)

        avg_init = torch.mean(init_img_features, axis=0)
        #avg_goal = torch.mean(goal_img_features, axis=0)
        #print(avg_init.shape, avg_goal.shape)

        self.delta_img_features = goal_img_features - avg_init.unsqueeze(0)



    def eval_img(self, im):
        im = self.preprocess(self.crop_img(im)).unsqueeze(0).to(device)
        if self.mode == 'clip': img_feature = self.model.encode_image(im) - self.base_features
        else: img_feature = self.model(im) - self.base_features
        similarity = np.mean(torch.nn.functional.cosine_similarity(img_feature,self.delta_img_features).cpu().numpy())
        return similarity



    def crop_img(self, im):
        # Setting the points for cropped image
        
        #left, right, top, bottom = 25, 150, 120, 248
        left, right, top, bottom = 5, 110, 110, 200
       
        '''if DEMO_MODE == 'microwave':
            left = 25
            right = 150
            top = 120 
            bottom = 248
        else:
            left = 65
            right = 195
            top = 15 
            bottom = 138'''
        
        im1 = im.crop((left, top, right, bottom))

        return im1
    



class KitchenBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS_AND_WEIGHTS = {
        "hand_jnt": 1.0,
        "objs_jnt": 1.0,
        "goal": 1.0,
        "goal_err": 1.0,
        "approach_err": 1.0,
    }
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "goal": 5.0,
        "bonus": 0.0, #0.5,
        "pose": 0.0, #0.01,
        "approach": 0.0, #0.5,
    }
    '''
    "goal": 1.0,
    "bonus": 0.5,
    "pose": 0.0, #0.01
    "approach": 0.5,
    '''

    def __init__(self, model_path, config_path,
                    robot_jnt_names, obj_jnt_names, obj_interaction_site,
                    goal, interact_site,
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

        # configure env-robot
        self.robot_dofs = []
        self.robot_ranges = []
        for jnt_name in robot_jnt_names:
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.robot_dofs.append(self.sim.model.jnt_dofadr[jnt_id])
            self.robot_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.robot_dofs = np.array(self.robot_dofs)
        self.robot_ranges = np.array(self.robot_ranges)
        self.robot_meanpos =  np.mean(self.robot_ranges,axis=1)

        # configure env-objs
        '''
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
        '''
        self.obj = {}
        obj_dof_adrs = []
        obj_dof_ranges = []
        for goal_adr, jnt_name in enumerate(obj_jnt_names):
            jnt_id = self.sim.model.joint_name2id(jnt_name)
            self.obj[jnt_name] = {}
            self.obj[jnt_name]['goal_adr'] = goal_adr
            self.obj[jnt_name]['interact_sid'] = self.sim.model.site_name2id(obj_interaction_site[goal_adr])
            self.obj[jnt_name]['dof_adr'] = self.sim.model.jnt_dofadr[jnt_id]
            obj_dof_adrs.append(self.sim.model.jnt_dofadr[jnt_id])
            obj_dof_ranges.append(self.sim.model.jnt_range[jnt_id])
        self.obj['dof_adrs'] = np.array(obj_dof_adrs)
        self.obj['dof_ranges'] = np.array(obj_dof_ranges)
        self.obj['dof_ranges'] = self.obj['dof_ranges'][:,1] - self.obj['dof_ranges'][:,0]

        # configure env-goal
        if interact_site == 'end_effector':
            print("WARNING: Using the default interaction site of end-effector. \
                  If you wish to evaluate on specific tasks, you should set the interaction site correctly.")
        self.set_goal(goal=goal, interact_site=interact_site)

        # increase simulation timestep for faster experiments
        # self.sim.model.opt.timestep = 0.008

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


        #self.deep_visual_reward_function = DeepFeatureSimilarityRewardFunction()

    

    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict['t'] = np.array([sim.data.time])

        ### raw observations
        obs_dict['hand_jnt'] = sim.data.qpos[self.robot_dofs].copy()
        obs_dict['objs_jnt'] = sim.data.qpos[self.obj['dof_adrs']].copy()
        obs_dict['hand_vel'] = sim.data.qvel[self.robot_dofs].copy() * self.dt
        obs_dict['objs_vel'] = sim.data.qvel[self.obj['dof_adrs']].copy() * self.dt
        obs_dict['goal'] = self.goal.copy()
        obs_dict['goal_err'] = obs_dict['goal']-obs_dict['objs_jnt'] # mix of translational and rotational erros
        obs_dict['approach_err'] = self.sim.data.site_xpos[self.interact_sid] - self.sim.data.site_xpos[self.grasp_sid]
        obs_dict['pose_err'] = self.robot_meanpos-obs_dict['hand_jnt']
        obs_dict['end_effector'] = self.sim.data.site_xpos[self.grasp_sid]
        for site in self.INTERACTION_SITES:
            site_id = self.sim.model.site_name2id(site)
            obs_dict[site+'_err'] = self.sim.data.site_xpos[site_id] - self.sim.data.site_xpos[self.grasp_sid]
        return obs_dict


    def get_reward_dict(self, obs_dict):
        
        try:
            visual_img = self.sim.render(width=256, height=256, depth=False, camera_name="overhead")
            img = Image.fromarray(cv2.flip(visual_img,0))
            visual_r = self.deep_visual_reward_function.eval_img(img)
        except Exception:
            self.deep_visual_reward_function = DeepFeatureSimilarityRewardFunction()
            visual_img = self.sim.render(width=256, height=256, depth=False, camera_name="overhead")
            img = Image.fromarray(cv2.flip(visual_img,0))
            visual_r = self.deep_visual_reward_function.eval_img(img)
        

        goal_dist = np.abs(obs_dict['goal_err'])

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('goal',    visual_r), #-np.sum(goal_dist, axis=-1)), # visual_r), #
            ('bonus',   np.product(goal_dist < 0.75*self.obj['dof_ranges'], axis=-1) + np.product(goal_dist < 0.25*self.obj['dof_ranges'], axis=-1)),
            ('pose',    -np.sum(np.abs(obs_dict['pose_err']), axis=-1)),
            ('approach',-np.linalg.norm(obs_dict['approach_err'], axis=-1)),
            # Must keys
            ('sparse',  visual_r), #-np.sum(goal_dist, axis=-1)), #visual_r ), #
            ('solved',  np.all(goal_dist < 0.15*self.obj['dof_ranges'])),
            ('done',    False),
        ))
        #print(goal_dist[0][0], 0.1*self.obj_ranges, np.all(goal_dist[0][0] < 0.1*self.obj_ranges))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        if self.mujoco_render_frames and VIZ:
            self.dict_plot.append(rwd_dict, self.rwd_keys_wt)
            # self.dict_plot.append(rwd_dict)

        return rwd_dict


    def set_goal(self, goal=None, interact_site=None):

        # resolve goals
        if type(goal) is dict:
            # treat current sim as goal
            self.goal = self.sim.data.qpos[self.obj['dof_adrs']].copy()
            # overwrite explicit requests
            for obj_name, obj_goal in goal.items():
                self.goal[self.obj[obj_name]['goal_adr']] = obj_goal
        elif type(goal) is np.ndarray:
            assert len(goal) == len(self.obj['dof_adrs']), "Check size of provided goal"
            self.goal = goal
        else:
            raise TypeError("goals must be either a dict<obj_name, obb_goal>, or a vector of all obj_goals")

        # resolve interaction site
        if interact_site is None: # automatically infer
            goal_err = np.abs(self.sim.data.qpos[self.obj['dof_adrs']] - self.goal)
            max_goal_err_obj = np.argmax(goal_err)
            for _,obj in self.obj.items():
                if obj['goal_adr'] == max_goal_err_obj:
                    self.interact_sid = obj['interact_sid']
                    break
        elif type(interact_site) is str: # overwrite using name
            self.interact_sid = self.sim.model.site_name2id(interact_site)
        elif type(interact_site) is int: # overwrite using id
            self.interact_sid = interact_site

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

class KitchenFrankaFixed(KitchenBase):

    INTERACTION_SITES = ['knob1_site', 'knob2_site', 'knob3_site', 'knob4_site','light_site', 'slide_site', 'leftdoor_site', 'rightdoor_site', 'microhandle_site', 'kettle_site0']

    def __init__(self,
                goal=None,
                interact_site="end_effector",
                **kwargs,
                ):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        KitchenBase.__init__(self,
            model_path = curr_dir + '/assets/franka_kitchen.xml',
            config_path = curr_dir + '/assets/franka_kitchen.config',
            robot_jnt_names = ('panda0_joint1', 'panda0_joint2', 'panda0_joint3', 'panda0_joint4', 'panda0_joint5', 'panda0_joint6', 'panda0_joint7', 'panda0_finger_joint1', 'panda0_finger_joint2'),
            obj_jnt_names = ('knob1_joint', 'knob2_joint', 'knob3_joint', 'knob4_joint', 'lightswitch_joint', 'slidedoor_joint', 'leftdoorhinge', 'rightdoorhinge', 'microjoint', 'kettle0:Tx', 'kettle0:Ty', 'kettle0:Tz', 'kettle0:Rx', 'kettle0:Ry', 'kettle0:Rz'),
            obj_interaction_site = tuple(self.INTERACTION_SITES+['kettle_site0', 'kettle_site0', 'kettle_site0', 'kettle_site0', 'kettle_site0']),
            goal=goal,
            interact_site=interact_site,
            **kwargs)


class KitchenFrankaRandom(KitchenFrankaFixed):
    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += 0.05*(self.np_random.uniform(size=len(self.robot_dofs))-0.5)*(self.robot_ranges[:,1] - self.robot_ranges[:,0])
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)
