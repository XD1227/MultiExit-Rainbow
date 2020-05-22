import io
import os
import datetime
import time
import math
import random
import numpy as np
import torch
import cv2
import copy
from xlwt import Workbook
import xlrd

from enum import Enum
import rvo.math as rvo_math
from rvo.vector import Vector2
from rvo.simulator import Simulator

'''
非常奇怪在opencv和numpy之间的格式转换，我们opencv=[width, height], 而numpy=[row, column]
width -> row
height -> column
其实是把图像做了一个转置
'''
Max_Speed = 3.0
Radius = 2.0
img_dim = 84
action_size = 8
width = 500  # 对应的是行
height = 500  # 对应的是列
cropped_pixel = int(min(width, height) / 4)

''' HyperParameters '''
w_g = 1.5
w_c = 0.2
w_d = -10


class Scenario(Enum):
    One_Exit = 0
    Two_Exits = 1


class Direction(Enum):
    Forward = 0
    Backward = 1
    FR = 2
    FL = 3
    BR = 4
    BL = 5
    Right = 6
    Left = 7
    Stop = 8


class Map_Screen(object):
    def __init__(self, map_min_x, map_max_x, map_min_y, map_max_y, screen_x, screen_y):
        self.update(map_min_x, map_max_x, map_min_y,
                    map_max_y, screen_x, screen_y)

    def update(self, map_min_x, map_max_x, map_min_y, map_max_y, screen_x, screen_y):
        self.map_minX = map_min_x
        self.map_maxX = map_max_x
        self.map_minY = map_min_y
        self.map_maxY = map_max_y
        self.map_h = map_max_y - map_min_y
        self.map_w = map_max_x - map_min_x
        self.win_x = screen_x
        self.win_y = screen_y
        self.scale_x = self.map_w / self.win_x
        self.scale_y = self.map_h / self.win_y

    def map_to_screen(self, point):
        screen_x = (point.x-self.map_minX)/self.scale_x
        screen_y = self.win_y - (point.y-self.map_minY)/self.scale_y
        if math.isnan(screen_x) or math.isnan(screen_y):
            print('ERROR:{} {}'.format(point.x, point.y))
            return [100, 100]
        return [int(screen_x), int(screen_y)]

    def screen_to_map(self, screen):
        map_x = self.scale_x*screen[0] + self.map_minX
        map_y = self.scale_y*(self.win_y-screen[1]) + self.map_minY
        return Vector2(map_x, map_y)


class PanicEnv(Simulator):
    '''
    由于是需要cv2画图，这是基于像素的，因此为了保持精度，倍数放大十倍
    '''

    def __init__(self, num_agents=30, scenario_=Scenario.One_Exit, open_time=0, load_agents=True, read_agents=False, action_repeat=4):
        super(PanicEnv, self).__init__()
        cv2.destroyAllWindows()
        self.clear_images()
        self.observation_space = (action_repeat, img_dim, img_dim)
        self.action_space = action_size
        self.goals = []
        self.arrived = [False for _ in range(num_agents)]
        self.d_prev = []
        self.v_prev = []
        self.colors = []
        self.scenario = scenario_
        global glo_num_agents
        glo_num_agents = num_agents
        self.setup_scenario(
            num_agents, load_agents=load_agents, read_agents=read_agents)
        self.cur_step = 0
        self.open_time = open_time

    def setup_scenario(self, num_agents, load_agents=False, read_agents=False):
        self.set_time_step(0.5)
        self.set_agent_defaults(Radius*5, 5, 5.0, 5.0, Radius,
                                Max_Speed, Vector2(0., 0.))
        extent = (-100., -100., 100., 100.)
        if self.scenario == Scenario.One_Exit:
            extent = self.init_single_exit_room(
                num_agents, load_agents=load_agents, read_agents=read_agents)
        elif self.scenario == Scenario.Two_Exits:
            extent = self.init_two_exit_room(
                num_agents, load_agents=load_agents, read_agents=read_agents)

        # view
        self.view = Map_Screen(
            extent[0], extent[2], extent[1], extent[3], width, height)

        # 转换坐标之后再画
        img = np.ones((width, height, 3), np.uint8) * 0
        img = self.draw_polygon(img)
        cv2.imwrite('screenshots/background.png', img)

    def reset(self, save_screenshots=False):
        # 1.初始化
        self.reset_room()
        self.clear_images()
        self.cur_step = 0
        # 2.走一步
        self.set_preferred_velocities()
        self.simulate()
        self.get_rewards()  # update the distance to destination
        # 3.获取状态
        self.get_states(save_screenshots)
        return self.states_ext  # (self.states_int, self.states_ext)

    def change_scenario(self):
        self.kd_tree_.obstacleTree_ = None
        if self.scenario == Scenario.Two_Exits:
            self.scenario = Scenario.One_Exit
            self.init_two_exit_room(glo_num_agents, load_agents=False)
        else:
            self.scenario = Scenario.Two_Exits
            self.init_two_exit_room(glo_num_agents, load_agents=False)

        img = np.ones((width, height, 3), np.uint8) * 0
        img = self.draw_polygon(img)
        cv2.imwrite('screenshots/background.png', img)

    def step(self, actions, save_screenshots=False, open_time=0):
        # 0.删除到达目的地的agent
        index = 0
        while index < self.num_agents:
            if self.arrived[index] is True:
                self.delete_agent(index)  # 可能报错吧
            else:
                index += 1
        num_agents = self.num_agents
        # 1.计算optimal速度
        for i in range(num_agents):
            if self.arrived[i] is True:
                continue
            pos = self.agents_[i].position_
            target = self.goals[i]
            g_heading = PanicEnv.action_to_vector_discrete(
                pos, target, actions[i], self.d_prev[i])
            self.set_agent_pref_velocity(i, g_heading)
        # 2.走一步
        self.simulate()
        # 3.获取状态，奖励
        end = self.reached_goal()
        self.get_states(save_screenshots)
        rewards = self.get_rewards()
        dones = self.arrived
        end = True if self.cur_step > 100 else end
        self.cur_step += 1
        if self.cur_step == self.open_time:
            self.change_scenario()
        # (self.states_int, self.states_ext)
        return self.states_ext, rewards, dones, end

    def init_single_exit_room(self, num_agents, width_=100, height_=100, load_agents=True, read_agents=False):

        inner_width = width_ * 0.8
        inner_height = height_ * 0.8
        wall_width = Radius * 1.5
        origin_eixt_width = (2.0 * Radius) * 2.0
        half_inner_width, half_inner_height = inner_width / 2., inner_height / 2.

        self.walls = []
        wall1 = []  # left top
        # wall2 = []  # left bottom
        wall3 = []  # bottom left
        wall4 = []  # bottom right
        wall5 = []  # top
        wall6 = []  # right

        wall1.append(Vector2(-half_inner_width -
                             wall_width, -half_inner_height))
        wall1.append(Vector2(-half_inner_width, -half_inner_height))
        wall1.append(Vector2(-half_inner_width, half_inner_height))
        wall1.append(Vector2(-half_inner_width-wall_width, half_inner_height))
        self.walls.append(wall1)

        half_bottom_exit_width = origin_eixt_width / 2.0
        wall3.append(Vector2(-half_inner_width-wall_width, -
                             half_inner_height-wall_width))
        wall3.append(Vector2(-half_bottom_exit_width, -
                             half_inner_height-wall_width))
        wall3.append(Vector2(-half_bottom_exit_width, -half_inner_height))
        wall3.append(Vector2(-half_inner_width-wall_width, -half_inner_height))
        self.walls.append(wall3)

        wall4.append(Vector2(half_bottom_exit_width, -
                             half_inner_height-wall_width))
        wall4.append(Vector2(half_inner_width+wall_width, -
                             half_inner_height-wall_width))
        wall4.append(Vector2(half_inner_width+wall_width, -half_inner_height))
        wall4.append(Vector2(half_bottom_exit_width, -half_inner_height))
        self.walls.append(wall4)

        wall5.append(Vector2(-half_inner_width-wall_width, half_inner_height))
        wall5.append(Vector2(half_inner_width+wall_width, half_inner_height))
        wall5.append(Vector2(half_inner_width+wall_width,
                             half_inner_height+wall_width))
        wall5.append(Vector2(-half_inner_width-wall_width,
                             half_inner_height+wall_width))
        self.walls.append(wall5)

        wall6.append(Vector2(half_inner_width, -half_inner_height))
        wall6.append(Vector2(half_inner_width+wall_width, -half_inner_height))
        wall6.append(Vector2(half_inner_width+wall_width, half_inner_height))
        wall6.append(Vector2(half_inner_width, half_inner_height))
        self.walls.append(wall6)

        self.add_obstacle(wall1)
        self.add_obstacle(wall3)
        self.add_obstacle(wall4)
        self.add_obstacle(wall5)
        self.add_obstacle(wall6)

        self.process_obstacles()

        self.exit1 = Vector2(0., -half_inner_height-wall_width-5.0)
        self.goals = [[self.exit1] for _ in range(num_agents)]

        if load_agents:
            if read_agents:
                self.read_agents_from_file('agents_info.xls')
            else:
                global glo_center, glo_border, glo_wall_width
                glo_center = [0., 0.]
                glo_border = [half_inner_width, half_inner_height]
                glo_wall_width = wall_width
                self.random_agents(num_agents, glo_center,
                                   glo_border, wall_width)

        map_min_x, map_min_y = -width_/2., -height_/2.
        map_max_x, map_max_y = width_/2., height_/2.
        return (map_min_x, map_min_y, map_max_x, map_max_y)

    def init_two_exit_room(self, num_agents, width_=100, height_=100, load_agents=True, read_agents=False):

        inner_width = width_ * 0.8
        inner_height = height_ * 0.8
        wall_width = Radius * 1.5
        origin_eixt_width = (2.0 * Radius) * 2.0
        # 1.0 is a hyperparameter, which to define the different of two exits
        another_exit_width = origin_eixt_width * 1.0
        half_inner_width, half_inner_height = inner_width / 2., inner_height / 2.

        self.walls = []
        wall1 = []  # left top
        wall2 = []  # left bottom
        wall3 = []  # bottom left
        wall4 = []  # bottom right
        wall5 = []  # top
        wall6 = []  # right

        half_left_exit_width = another_exit_width / 2.0
        wall1.append(Vector2(-half_inner_width -
                             wall_width, half_left_exit_width))
        wall1.append(Vector2(-half_inner_width, half_left_exit_width))
        wall1.append(Vector2(-half_inner_width, half_inner_height))
        wall1.append(Vector2(-half_inner_width-wall_width, half_inner_height))
        self.walls.append(wall1)

        wall2.append(Vector2(-half_inner_width-wall_width, -half_inner_height))
        wall2.append(Vector2(-half_inner_width, -half_inner_height))
        wall2.append(Vector2(-half_inner_width, -half_left_exit_width))
        wall2.append(Vector2(-half_inner_width -
                             wall_width, -half_left_exit_width))
        self.walls.append(wall2)

        half_bottom_exit_width = origin_eixt_width / 2.0
        wall3.append(Vector2(-half_inner_width-wall_width, -
                             half_inner_height-wall_width))
        wall3.append(Vector2(-half_bottom_exit_width, -
                             half_inner_height-wall_width))
        wall3.append(Vector2(-half_bottom_exit_width, -half_inner_height))
        wall3.append(Vector2(-half_inner_width-wall_width, -half_inner_height))
        self.walls.append(wall3)

        wall4.append(Vector2(half_bottom_exit_width, -
                             half_inner_height-wall_width))
        wall4.append(Vector2(half_inner_width+wall_width, -
                             half_inner_height-wall_width))
        wall4.append(Vector2(half_inner_width+wall_width, -half_inner_height))
        wall4.append(Vector2(half_bottom_exit_width, -half_inner_height))
        self.walls.append(wall4)

        wall5.append(Vector2(-half_inner_width-wall_width, half_inner_height))
        wall5.append(Vector2(half_inner_width+wall_width, half_inner_height))
        wall5.append(Vector2(half_inner_width+wall_width,
                             half_inner_height+wall_width))
        wall5.append(Vector2(-half_inner_width-wall_width,
                             half_inner_height+wall_width))
        self.walls.append(wall5)

        wall6.append(Vector2(half_inner_width, -half_inner_height))
        wall6.append(Vector2(half_inner_width+wall_width, -half_inner_height))
        wall6.append(Vector2(half_inner_width+wall_width, half_inner_height))
        wall6.append(Vector2(half_inner_width, half_inner_height))
        self.walls.append(wall6)

        self.add_obstacle(wall1)
        self.add_obstacle(wall2)
        self.add_obstacle(wall3)
        self.add_obstacle(wall4)
        self.add_obstacle(wall5)
        self.add_obstacle(wall6)

        self.process_obstacles()

        self.exit1 = Vector2(0., -half_inner_height-wall_width-5.0)
        self.exit2 = Vector2(-half_inner_width-wall_width-5.0, 0.)
        self.goals = [[self.exit1, self.exit2] for _ in range(num_agents)]

        if load_agents:
            if read_agents:
                self.read_agents_from_file('agents_info.xls')
            else:
                global glo_center, glo_border, glo_wall_width
                glo_center = [0., 0.]
                glo_border = [half_inner_width, half_inner_height]
                glo_wall_width = wall_width
                self.random_agents(num_agents, glo_center,
                                   glo_border, wall_width)

        map_min_x, map_min_y = -width_/2., -height_/2.
        map_max_x, map_max_y = width_/2., height_/2.
        return (map_min_x, map_min_y, map_max_x, map_max_y)

    def random_agents(self, num_agents, center, border, w_w, ratio=[1, 1]):
        self.coordinates = []
        self.max_speeds = np.random.random(num_agents) * 3. + Max_Speed
        exit_1 = (int)(ratio[0] * num_agents / sum(ratio))
        # (int)(ratio[1] * num_agents / sum(ratio))
        exit_2 = num_agents - exit_1
        exits_count = [exit_1, exit_2]
        ratio_count = [0, 0]
        i = 0
        while i < (exit_1+exit_2):
            coord = center[0]-border[0]+w_w*3 + \
                np.random.random(2) * (border[0]*2-w_w*6)
            pos = Vector2(coord[0], coord[1])
            dists = self.dist_to_goals(pos, self.goals[i])
            dists = np.array(dists)
            min_index = dists.argmin()
            if ratio_count[min_index] < exits_count[min_index]:
                ratio_count[min_index] += 1
                self.add_agent(pos)
                self.colors.append(np.random.randint(0, 255, 3))
                self.agents_[i].max_speed_ = self.max_speeds[i]
                self.agents_[i].color_ = self.colors[i]
                self.d_prev.append(dists)
                self.v_prev.append(Vector2(0., 0.))
                self.coordinates.append(coord[0])
                self.coordinates.append(coord[1])
                i += 1
        self.save_agents('agents_info.xls')

    def dist_to_goals(self, pos, goals):
        dists = []
        for i in range(len(goals)):
            goal = goals[i]
            dist = math.sqrt(rvo_math.abs_sq(goal-pos))
            dists.append(dist)
        return dists

    def reset_room(self):
        index = 0
        while index < self.num_agents:
            self.delete_agent(index)

        if self.open_time == 0:
            self.goals = [[self.exit1] for _ in range(glo_num_agents)]
        else:
            self.goals = [[self.exit1, self.exit2]
                          for _ in range(glo_num_agents)]
        self.global_time_ = 0
        self.kd_tree_.agents_ = None
        self.kd_tree_.agentTree_ = None
        self.d_prev = []
        self.v_prev = []
        # self.arrived = [False for _ in range(glo_num_agents)]
        # self.random_agents(glo_num_agents, glo_center, glo_border, glo_wall_width)
        for i in range(glo_num_agents):
            pos = Vector2(self.coordinates[i*2], self.coordinates[i*2+1])
            self.add_agent(pos)
            self.agents_[i].max_speed_ = Max_Speed  # self.max_speeds[i]
            self.agents_[i].color_ = self.colors[i]
            self.arrived.append(False)
            dists = self.dist_to_goals(pos, self.goals[i])
            self.d_prev.append(dists)
            self.v_prev.append(Vector2(0., 0.))

    def set_preferred_velocities(self):
        for i in range(self.num_agents):
            pos = self.agents_[i].position_
            target = self.goals[i][0]
            g_heading = rvo_math.normalize(target - pos) * Max_Speed
            dists = self.dist_to_goals(pos, self.goals[i])
            self.d_prev[i] = dists
            self.v_prev[i] = self.agents_[i].velocity_
            self.set_agent_pref_velocity(i, g_heading)

    def reached_goal(self):
        total_reached = True
        index = 0
        while index < self.num_agents:
            agent = self.agents_[index]
            # method 2
            pos = agent.position_
            in_room = pos.x > -40 and pos.x < 40 and pos.y > -40 and pos.y < 40
            in_room = bool(in_room) if isinstance(
                in_room, np.bool_) else in_room

            if in_room is True:
                self.arrived[index] = False
                total_reached = False
            else:
                self.arrived[index] = True
            # method 1
            # min_dist = self.min_dist_to_goal(agent.position_, self.goals[index])
            # if min_dist > 4.0 * self.agents_[index].radius_:
            #     self.arrived[index] = False
            # else:
            #     self.arrived[index] = True
            # if min_dist > 4.0 * Radius and total_reached is True:
            #     total_reached = False
            index += 1
        return total_reached

    def min_dist_to_goal(self, pos, goals):
        min_dist = 1e+5
        dists = self.dist_to_goals(pos, goals)

        for i in range(len(dists)):
            if dists[i] < min_dist:
                min_dist = dists[i]
        return min_dist

    def simulate(self):
        self.step_(self.arrived)

    def get_states(self, save_screenshots=False):
        self.states_ext = []
        self.states_int = []
        img_1 = cv2.imread('screenshots/background.png')

        binary_img = self.draw(img_1, binary_=True)
        if save_screenshots is True:
            img_2 = cv2.imread('screenshots/background.png')
            # img_2 = 255 - img_2
            rbg_img = self.draw(img_2, binary_=False)
            cv2.imwrite('screenshots/simulation/frame' +
                        str(self.cur_step)+'.png', rbg_img)
        for i in range(self.num_agents):
            target_img = copy.deepcopy(binary_img)
            target_img = self.draw_circle(target_img, i, True, (250, 250, 250))

            target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
            target_img = cv2.resize(
                target_img, (img_dim, img_dim), interpolation=cv2.INTER_AREA)
            # cv2.imwrite('screenshots/simulation/frame' +
            #             str(self.cur_step)+'-'+str(i)+'.png', target_img)
            self.states_ext.append(target_img)
        self.states_ext = np.array(self.states_ext)
        self.states_ext = (self.states_ext - self.states_ext.mean()
                           ) / (self.states_ext.std() + 1e-5)
        self.states_int = np.array(self.states_int)
        del img_1
        del binary_img
        if save_screenshots is True:
            del img_2
            del rbg_img

    def cal_addict_rewards(self, curr, prev, important_ratio):

        curr, prev = np.array(curr), np.array(prev)
        important_curr = curr / important_ratio
        important_prev = prev / important_ratio
        addict_reward_curr = 1.0 - (important_curr)**0.4
        addict_reward_prev = 1.0 - (important_prev)**0.4

        reward = (addict_reward_curr.max() - addict_reward_prev.max()) * 15

        return reward

    def cal_collision_reward(self, pos, goals, vel, d_cur):
        d_cur_np = np.array(d_cur)

        rewards = []
        for goal in goals:
            direct = rvo_math.normalize(goal-pos)
            reward = (direct.x*vel.x + direct.y*vel.y)/Max_Speed
            rewards.append(reward)
        pos_count = np.sum(list(map(lambda x: x >= 0, rewards)))
        if pos_count.item() > 1:
            important_ratio = d_cur_np.max() / d_cur_np
            important_ratio = important_ratio / important_ratio.max()
            rewards = important_ratio * np.array(rewards)
        elif pos_count.item() == 1:
            return np.array(rewards).max()
        else:
            important_ratio = d_cur_np / d_cur_np.max()
            rewards = important_ratio * np.array(rewards)
        return rewards.max()

    def cal_smooth_rewards(self, v_prev, v_cur):
        v_prev_ = rvo_math.normalize(v_prev)
        v_cur_ = rvo_math.normalize(v_cur)
        dot_product = v_prev_.x * v_cur_.x + v_prev_.y * v_cur_.y
        reward = dot_product * 1.0
        return reward

    def get_rewards(self):
        rewards = [None for _ in range(self.num_agents)]
        important_ratio = np.array([1.0, 1.0])
        w1, w2, w3, w4 = 1.0, 1.25, 0.5, 2.5
        for i in range(self.num_agents):
            agent = self.agents_[i]
            pos = agent.position_
            d_cur = self.dist_to_goals(pos, self.goals[i])
            d_prev = self.d_prev[i]
            v_prev = self.v_prev[i]

            # rewards function
            addict_reward = self.cal_addict_rewards(
                d_cur, d_prev, important_ratio)
            collision_reward = self.cal_collision_reward(
                agent.position_, self.goals[i], agent.velocity_, d_cur)
            smooth_reward = self.cal_smooth_rewards(v_prev, agent.velocity_)
            rewards[i] = w1*addict_reward + w2 * \
                collision_reward + w3*smooth_reward - w4
            self.d_prev[i] = d_cur
            self.v_prev[i] = agent.velocity_
        return rewards

    def get_rewards_old(self):
        rewards = [None for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            agent = self.agents_[i]
            pos = agent.position_
            d_cur = self.dist_to_goals(pos, self.goals[i])
            d_prev = self.d_prev[i]
            v_prev = self.v_prev[i]

            # rewards function
            addict_reward, _ = self.cal_addict_rewards(d_cur, d_prev)
            collision_reward = self.cal_collision_reward(agent.velocity_)
            smooth_reward = self.cal_smooth_rewards(v_prev, agent.velocity_)
            rewards[i] = addict_reward + collision_reward + smooth_reward - 3.
            self.d_prev[i] = d_cur
            self.v_prev[i] = agent.velocity_
        return rewards

    def delete_agent(self, i):
        # 根据索引，需要删除 agents_, goals_, d_prev, arrived
        del self.agents_[i]
        del self.goals[i]
        del self.d_prev[i]
        del self.v_prev[i]
        del self.arrived[i]

    def draw(self, img, binary_=False):
        if self.scenario == Scenario.One_Exit or self.scenario == Scenario.Two_Exits:
            for i in range(self.num_agents):
                img = self.draw_circle(img, i, binary_=binary_)
        return img

    def draw_circle(self, img, index, binary_=False, color=(100, 100, 100)):
        agent = self.agents_[index]
        color = agent.color_.tolist() if binary_ is False else color
        pos = agent.position_
        radius = agent.radius_ / self.view.scale_x

        pos = self.view.map_to_screen(pos)
        cv2.circle(img, (pos[0], pos[1]), int(radius), color, -1)
        return img

    def draw_polygon(self, img):
        for wall in self.walls:
            pts = []
            for p in wall:
                pts.append(self.view.map_to_screen(p))

            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (100, 100, 100))

        return img

    def clear_images(self):
        path = '/screenshots/simulation'
        path = os.getcwd() + path
        num_imgs = len(os.listdir(path))
        for i in range(num_imgs):
            filename = path + '/frame'+str(i)+'.png'
            if os.path.exists(filename):
                os.remove(filename)

    def save_agents(self, filename):
        xldoc = Workbook()
        sheet1 = xldoc.add_sheet('Agents_Info', cell_overwrite_ok=True)
        sheet1.write(0, 0, 'Agent')
        sheet1.write(0, 1, 'Position')
        sheet1.write(0, 2, 'Color')
        sheet1.write(0, 3, 'Max_Speed')
        for i in range(len(self.colors)):
            pos = str(self.coordinates[i*2]) + \
                ',' + str(self.coordinates[i*2+1])
            color = str(self.colors[i][0]) + ',' + \
                str(self.colors[i][1]) + ',' + str(self.colors[i][2])
            max_speed = str(self.max_speeds[i])
            sheet1.write(i+1, 0, str(i))
            sheet1.write(i+1, 1, pos)
            sheet1.write(i+1, 2, color)
            sheet1.write(i+1, 3, max_speed)
        xldoc.save(filename)

    def read_agents_from_file(self, filename):

        wb = xlrd.open_workbook(filename)
        sheet = wb.sheet_by_index(0)

        self.coordinates = []
        self.colors = []
        self.max_speeds = []
        for i in range(1, sheet.nrows):
            # agent_id = sheet.cell_value(i, 0)
            pos = sheet.cell_value(i, 1).split(',')
            color = sheet.cell_value(i, 2).split(',')
            max_speed = float(sheet.cell_value(i, 3))

            pos_x, pos_y = float(pos[0]), float(pos[1])
            pos = Vector2(pos_x, pos_y)
            color = torch.tensor([int(color[0]), int(color[1]), int(color[2])])
            self.add_agent(pos)
            self.agents_[i-1].max_speed_ = max_speed
            self.agents_[i-1].color_ = np.array(color)
            self.coordinates.append(pos_x)
            self.coordinates.append(pos_y)
            self.colors.append(color)
            self.max_speeds.append(max_speed)
            dists = self.dist_to_goals(pos, self.goals[i-1])
            self.d_prev.append(dists)
            self.v_prev.append(Vector2(0., 0.))
        print('load done')

    @staticmethod
    def action_to_vector(pos, target, action, max_speed, ORCA=False):
        angle, speed = action[0], max_speed
        heading = rvo_math.normalize(target-pos)
        # if ORCA:
        #     heading = rvo_math.normalize(target-pos)
        # else:
        #     heading = Vector2(1., 0.)  # rvo_math.normalize(target-pos)
        # angle = -math.pi + random.random() * 2 * math.pi
        x1, y1 = heading.x, heading.y
        x2 = math.cos(angle) * x1 - math.sin(angle) * y1
        y2 = math.sin(angle) * x1 + math.cos(angle) * y1
        return speed * Vector2(x2, y2)

    @staticmethod
    def action_to_vector_discrete(pos, target, direction, prev):
        # go straight if near to the goal
        # min_dist = np.array(prev).min()
        # if min_dist < Radius * 8:
        #     min_index = np.array(prev).argmax()
        #     directV = rvo_math.normalize(target[min_index]-pos)
        #     return directV * Max_Speed
        V_pref = Vector2(0.0, 0.0)
        # rvo_math.normalize(target[0]-pos)  # Vector2(0., 1.)  #
        directV = Vector2(0., 1.)
        if direction == Direction.Forward.value:
            V_pref = directV
        elif direction == Direction.Backward.value:
            V_pref = -directV
        elif direction == Direction.FR.value:
            theta = -math.pi / 4.
            x1, y1 = directV.x, directV.y
            x2 = math.cos(theta) * x1 - math.sin(theta) * y1
            y2 = math.sin(theta) * x1 + math.cos(theta) * y1
            V_pref = Vector2(x2, y2)
        elif direction == Direction.FL.value:
            theta = math.pi / 4
            x1, y1 = directV.x, directV.y
            x2 = math.cos(theta) * x1 - math.sin(theta) * y1
            y2 = math.sin(theta) * x1 + math.cos(theta) * y1
            V_pref = Vector2(x2, y2)
        elif direction == Direction.BR.value:
            theta = -3*math.pi / 4.
            x1, y1 = directV.x, directV.y
            x2 = math.cos(theta) * x1 - math.sin(theta) * y1
            y2 = math.sin(theta) * x1 + math.cos(theta) * y1
            V_pref = Vector2(x2, y2)
        elif direction == Direction.BL.value:
            theta = 3*math.pi / 4
            x1, y1 = directV.x, directV.y
            x2 = math.cos(theta) * x1 - math.sin(theta) * y1
            y2 = math.sin(theta) * x1 + math.cos(theta) * y1
            V_pref = Vector2(x2, y2)
        elif direction == Direction.Right.value:
            theta = -math.pi / 2.
            x1, y1 = directV.x, directV.y
            x2 = math.cos(theta) * x1 - math.sin(theta) * y1
            y2 = math.sin(theta) * x1 + math.cos(theta) * y1
            V_pref = Vector2(x2, y2)
        elif direction == Direction.Left.value:
            theta = math.pi / 2
            x1, y1 = directV.x, directV.y
            x2 = math.cos(theta) * x1 - math.sin(theta) * y1
            y2 = math.sin(theta) * x1 + math.cos(theta) * y1
            V_pref = Vector2(x2, y2)
        else:
            V_pref = Vector2(0., 0.)
        return V_pref * Max_Speed

    @staticmethod
    def display(save_as_video=False):
        path = '/screenshots/simulation'
        path = os.getcwd() + path
        num_imgs = len(os.listdir(path))
        cv2.namedWindow('Simulation', cv2.WINDOW_NORMAL)
        if save_as_video:
            # avi_cc = cv2.VideoWriter_fourcc('M','J','P','G')
            mp4_cc = 0x7634706d
            path_video = os.getcwd() + '/screenshots/videos/PanicSim-{}.mp4'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            out = cv2.VideoWriter(path_video, mp4_cc, 8, (width, height))

        for i in range(num_imgs):
            filename = path + '/frame'+str(i)+'.png'
            if os.path.exists(filename):
                img = cv2.imread(filename)
                cv2.imshow('Simulation', img)
                if save_as_video:
                    out.write(img)
                time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                break
        if save_as_video:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    env = PanicEnv()
