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
import random

import clip

VIZ = False
device = "cuda" if torch.cuda.is_available() else "cpu"
DEMO_MODE = 'microwave' #'cabinet' #
#BASE_DIR = '/Users/yuchencui/Projects/active_learning/'
BASE_DIR = '/private/home/yuchencui/projects/active_learning/'

class DeepFeatureSimilarityRewardFunction(object):
    def __init__(self, task_id ,mode ='clip') -> None:
        super().__init__()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        
        env_site_dict = {
            "knob1_site" : "kitchen_knob1_on-v3",
            "knob2_site" : "kitchen_knob2_on-v3",
            "knob3_site" : "kitchen_knob3_on-v3",
            "knob4_site" : "kitchen_knob4_on-v3",
            "leftdoor_site": "kitchen_ldoor_open-v3",
            "light_site": "kitchen_light_on-v3",
            "microhandle_site" : "kitchen_micro_open-v3",
            "rightdoor_site" : "kitchen_rdoor_open-v3",
            "slide_site" : "kitchen_sdoor_open-v3"  
         }

        self.task_id = env_site_dict[task_id]
        if 'micro' in task_id or 'leftdoor' in task_id or 'rightdoor' in task_id:
            self.view_points = ['eye_level', 'eye_level1', 'eye_level2', 'eye_level3', 'eye_level4']
        elif 'knob' in task_id or 'light' in task_id:
            self.view_points = ['counter_top', 'counter_top1', 'counter_top2', 'counter_top3', 'counter_top4']
        else:
            self.view_points = ['overhead', 'overhead1', 'overhead2', 'overhead3', 'overhead4']
        #print(task_id, self.task_id, self.view_points)


        self.debug_ct = 0

        with torch.no_grad():
            if mode == 'clip': 
                self.model, self.preprocess = clip.load("ViT-B/32", device=device)
            else: 
                self.model = models.resnet152(pretrained=True)
                self.model = self.model.to(device)
        
        self.mode = mode
        modules=list(self.model.children())[:-1]
        self.modelmodel=torch.nn.Sequential(*modules)
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.model.eval()

        data_dir = BASE_DIR + 'online_imgs/kitchen_micro_open-v3/'

        goal_img_paths = [os.path.join(data_dir+'/goal', f) for f in os.listdir(data_dir+'/goal') if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.webp')] 
        init_img_paths = [os.path.join(data_dir+'/init', f) for f in os.listdir(data_dir+'/init') if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.webp')] 

        if len(goal_img_paths) > 25: goal_img_paths = random.sample(goal_img_paths,25)
        if len(init_img_paths) > 25: init_img_paths = random.sample(init_img_paths,25)

        goal_imgs = torch.stack([self.preprocess(Image.open(f)).to(device) for f in goal_img_paths])
        init_imgs = torch.stack([self.preprocess(Image.open(f)).to(device) for f in init_img_paths])

        base_img_dir = BASE_DIR + "/base_imgs/"
        self.base_imgs = None  #[self.crop_img(Image.open(base_img_dir+env_site_dict[task_id]+'/'+viewpoint+'/base_img.png'),view_point=viewpoint) for viewpoint in self.view_points]
        #base_images = [self.preprocess(self.crop_img(Image.open(base_img_dir+env_site_dict[task_id]+'/'+viewpoint+'/base_img.png'),view_point=viewpoint)) for viewpoint in self.view_points]
        #base_image_tensor = torch.stack(base_images).to(device)

        if self.mode == 'clip':
            goal_img_features = self.model.encode_image(goal_imgs)
            init_img_features = self.model.encode_image(init_imgs) 
            #self.base_features = self.model.encode_image(base_image_tensor)
        else:
            goal_img_features = self.model(goal_imgs)
            init_img_features = self.model(init_imgs)
            #self.base_features = self.model(base_image_tensor)
        self.base_features = None

        #avg_init = torch.mean(init_img_features, axis=0)
        #self.delta_img_features = goal_img_features - avg_init.unsqueeze(0)

        delta_img_features = []
        for g_img in goal_img_features:
            highest_similarity = - np.inf
            mll_delta_feature = None
            for i_img in init_img_features:
                sim = torch.nn.functional.cosine_similarity(g_img.unsqueeze(0),i_img.unsqueeze(0)).cpu()
                if sim > highest_similarity:
                    highest_similarity = sim
                    mll_delta_feature = g_img.cpu().numpy() - i_img.cpu().numpy()
            delta_img_features.append(mll_delta_feature)

        self.delta_img_features = torch.tensor(delta_img_features).to(device)



    def eval_img(self, im):
        im = self.preprocess(self.crop_img(im)).unsqueeze(0).to(device)
        if self.mode == 'clip': img_feature = self.model.encode_image(im) - self.base_features[0]
        else: img_feature = self.model(im) - self.base_features[0]
        similarity = np.mean(torch.nn.functional.cosine_similarity(img_feature,self.delta_img_features).cpu().numpy())
        return similarity


    def eval_imgs(self, ims):
        processed_imgs = []
        self.debug_ct += 1
        for im_i in range(len(ims)):
            img = self.crop_img(ims[im_i],view_point=self.view_points[im_i])
            #print('saving img',self.view_points[im_i])
            '''
            delta_img = Image.fromarray(np.array(img) - np.array(self.base_imgs[im_i]))
            if np.sum(delta_img)>0:
                img_path = "/private/home/yuchencui/projects/active_learning/franka_baselines/imgs/debug_delta_"+str(self.view_points[im_i])+'_'+str(self.debug_ct)+".png"
                delta_img.save(img_path)
            '''
            processed_imgs.append(self.preprocess(img))
            
        ims = torch.stack(processed_imgs).to(device)
        if self.mode == 'clip': img_feature = self.model.encode_image(ims) - self.base_features
        else: img_feature = self.model(ims) - self.base_features
        ## average over each viewpoint, compute feature similarity avg delta feature
        all_similarities = []
        for im_i in range(len(ims)):
            if torch.norm(img_feature[im_i,:]) > 0.09:
                #print(im_i, torch.norm(img_feature[im_i,:]))
                similarities = torch.nn.functional.cosine_similarity(img_feature[im_i,:].unsqueeze(0), self.delta_img_features)
                similarity = np.mean(similarities.cpu().numpy())
                if similarity > 0.02: all_similarities.append(similarity)
                else: all_similarities.append(0.0)
            else:
                all_similarities.append(0.0)
        #print(self.debug_ct, all_similarities)
        similarity = np.mean(all_similarities)
        return similarity


    def crop_img(self, im, view_point):
        if view_point is None:
            if 'micro' in self.task_id:
                #left, right, top, bottom = 25, 150, 120, 248 # eye-level view crop
                left, right, top, bottom = 5, 110, 110, 200 # overhead view crop
            elif 'ldoor' in self.task_id  or 'rdoor' in self.task_id:
                #left, right, top, bottom = 65, 195, 25, 100
                left, right, top, bottom = 5, 135, 5, 80
            elif 'knob' in self.task_id:
                left, right, top, bottom = 90, 205, 80, 200
            elif 'sdoor' in self.task_id:
                #left, right, top, bottom = 180, 256, 30, 110
                left, right, top, bottom = 130, 245, 5, 80
            else:
                left, right, top, bottom = 5, 251, 5, 241
        elif 'micro' in self.task_id:
            if view_point == 'eye_level1':
                left, right, top, bottom = 0, 120, 140, 248 # eye-level view crop
            elif view_point == 'eye_level2':
                left, right, top, bottom = 5, 125, 140, 252 # eye-level view crop
            elif view_point == 'eye_level3':
                left, right, top, bottom = 25, 150, 100, 200 # eye-level view crop
            elif view_point == 'eye_level4':
                left, right, top, bottom = 25, 160, 120, 248 # eye-level view crop
            else:
                left, right, top, bottom = 25, 160, 120, 248 # eye-level view crop
            #left, right, top, bottom = 5, 110, 110, 200 # overhead view crop
        elif 'ldoor' in self.task_id  or 'rdoor' in self.task_id:
            if view_point == 'eye_level1':
                left, right, top, bottom = 15, 165, 25, 100
            elif view_point == 'eye_level2':
                left, right, top, bottom = 25, 165, 25, 100
            elif view_point == 'eye_level3':
                left, right, top, bottom = 55, 200, 0, 60
            elif view_point == 'eye_level4':
                left, right, top, bottom = 50, 200, 25, 90
            else:
                left, right, top, bottom = 55, 195, 25, 105
            #left, right, top, bottom = 5, 135, 5, 80
        #elif 'knob' in self.task_id:
        #    left, right, top, bottom = 90, 205, 80, 200
        elif 'sdoor' in self.task_id:
            if view_point == 'overhead1':
                left, right, top, bottom = 130, 250, 0, 50
            elif view_point == 'overhead2':
                left, right, top, bottom = 130, 250, 20, 100
            elif view_point == 'overhead3':
                left, right, top, bottom = 120, 245, 5, 80
            elif view_point == 'overhead4':
                left, right, top, bottom = 130, 245, 5, 80
            else:
                left, right, top, bottom = 135, 255, 5, 80
        else:
            left, right, top, bottom = 5, 251, 5, 241
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
        "approach": 0.1, #0.5,
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

        #print(config_path)
        site_id = self.sim.model.site_id2name(self.interact_sid)
        self.deep_visual_reward_function = DeepFeatureSimilarityRewardFunction(site_id)
        self.similarities = []
        if self.deep_visual_reward_function.base_imgs is None:
            imgs = []
            for viewpoint in self.deep_visual_reward_function.view_points:
                visual_img = self.sim.render(width=256, height=256, depth=False, camera_name=viewpoint)
                img = Image.fromarray(cv2.flip(visual_img,0))
                imgs.append(img)
            base_images = [self.deep_visual_reward_function.preprocess(self.deep_visual_reward_function.crop_img(imgs[i],view_point=self.deep_visual_reward_function.view_points[i])) for i in range(len(imgs))]
            self.deep_visual_reward_function.base_imgs = [self.deep_visual_reward_function.crop_img(imgs[i],view_point=self.deep_visual_reward_function.view_points[i]) for i in range(len(imgs))]
            base_image_tensor = torch.stack(base_images).to(device)
            if self.deep_visual_reward_function.mode == 'clip':
                self.deep_visual_reward_function.base_features = self.deep_visual_reward_function.model.encode_image(base_image_tensor)
            else:
                self.deep_visual_reward_function.base_features = self.deep_visual_reward_function.model(base_image_tensor)
        #print('init')

    

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
        site_id = self.sim.model.site_id2name(self.interact_sid)
        try:
            imgs = []
            for viewpoint in self.deep_visual_reward_function.view_points:
                visual_img = self.sim.render(width=256, height=256, depth=False, camera_name=viewpoint)
                img = Image.fromarray(cv2.flip(visual_img,0))
                imgs.append(img)
        except Exception as e:
            print('initializing sim detector')
            self.deep_visual_reward_function = DeepFeatureSimilarityRewardFunction(site_id)
            self.similarities = []
            imgs = []
            for viewpoint in self.deep_visual_reward_function.view_points:
                visual_img = self.sim.render(width=256, height=256, depth=False, camera_name=viewpoint)
                img = Image.fromarray(cv2.flip(visual_img,0))
                imgs.append(img)

        if self.deep_visual_reward_function.base_imgs is None:
            print('initializing base images')
            base_images = [self.deep_visual_reward_function.preprocess(self.deep_visual_reward_function.crop_img(imgs[i],view_point=self.deep_visual_reward_function.view_points[i])) for i in range(len(imgs))]
            self.deep_visual_reward_function.base_imgs = [self.deep_visual_reward_function.crop_img(imgs[i],view_point=self.deep_visual_reward_function.view_points[i]) for i in range(len(imgs))]
            base_image_tensor = torch.stack(base_images).to(device)
            if self.deep_visual_reward_function.mode == 'clip':
                self.deep_visual_reward_function.base_features = self.deep_visual_reward_function.model.encode_image(base_image_tensor)
            else:
                self.deep_visual_reward_function.base_features = self.deep_visual_reward_function.model(base_image_tensor)
            im_i = 0
            for img in imgs:
                img.save('/private/home/yuchencui/projects/active_learning/franka_baselines/imgs/debug_base_'+self.deep_visual_reward_function.view_points[im_i]+'.png')
                im_i += 1

        v_r = self.deep_visual_reward_function.eval_imgs(imgs)
        self.similarities.append(v_r)
        if len(self.similarities) > 3: del self.similarities[0]
        visual_r = np.mean(self.similarities)
        if visual_r < 0.022: visual_r = 0.0
        visual_r *= 10.0

        goal_dist = np.abs(obs_dict['goal_err'])
        dense_r = -np.sum(goal_dist, axis=-1)
        if visual_r > 0:
            print('visual reward: ',visual_r, 'goal reward:', dense_r, 'approach:', -0.1*np.linalg.norm(obs_dict['approach_err'], axis=-1))
            im_i = 0
            for img in imgs:
                img.save('/private/home/yuchencui/projects/active_learning/franka_baselines/imgs/debug_pos_sim_'+self.deep_visual_reward_function.view_points[im_i]+'.png')
                im_i += 1

        visual_r = np.array([[visual_r]])

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            #('goal',    -np.sum(goal_dist, axis=-1)), # visual_r), #
            ('goal',    visual_r), #
            ('bonus',   np.product(goal_dist < 0.75*self.obj['dof_ranges'], axis=-1) + np.product(goal_dist < 0.25*self.obj['dof_ranges'], axis=-1)),
            ('pose',    -np.sum(np.abs(obs_dict['pose_err']), axis=-1)),
            ('approach',-np.linalg.norm(obs_dict['approach_err'], axis=-1)),
            # Must keys
            ('sparse',  visual_r), #
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
