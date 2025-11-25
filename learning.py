import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd


# Q-learning settings
learning_rate = 0.005
discount_factor = 0.99
train_epochs = 1
learning_steps_per_epoch = 100000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 10

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 50

save_model = True
load_model = False
skip_learning = False

# Configuration file path
file_path_cfg = "defend_the_center.cfg" #"deathmatch.cfg" #"simpler_basic.cfg" # docs/scenarios_path.txt
config_file_path = os.path.join(vzd.scenarios_path, file_path_cfg) 


model_savefile = "./model-doom-{}.pth".format(file_path_cfg.replace(".cfg", ""))


# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)
print("Device name:", torch.cuda.get_device_name(device=DEVICE))

###############################################################################################################################

def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    return img


def create_simple_game():
    print("\nInitializing doom...")

    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8) # todo, verificar com cores
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    
    print("\nTesting...")

    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()

        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)

        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)

    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()


    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0

        print(f"\nEpoch #{epoch + 1}")

        for _ in trange(steps_per_epoch, leave=False):

            game_state = game.get_state()
            state = preprocess(game_state.screen_buffer) ## pega a tela do jogo e transforma em escala de cinza e redimensiona
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat) - game_state.game_variables[0] * 5 # -0.01 for using ammo
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        # print(
        #     "Results: mean: {:.1f} +/- {:.1f},".format(
        #         train_scores.mean(), # media - valor medio dos scores
        #         train_scores.std() # desvio padrao - quanto os valores se afastam da media
        #     ),

        #     "min: %.1f," % train_scores.min(), # minimo - menor valor dos scores
        #     "max: %.1f," % train_scores.max(), # maximo - maior valor dos scores
        # )

        # test(game, agent) # todo, ver se faz sentido testar a cada epoca

        
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    if save_model:
        print("Saving the network weights to:", model_savefile)
        torch.save(agent.q_net, model_savefile)

    test(game, agent)

    game.close()
    return agent, game


class DuelQNet(nn.Module):
    def __init__(self, available_actions_count):
        super().__init__()
        # 1 -> 8 -> 8 -> 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),# uma camada convolucional com 8 canais de entrada e saída, kernel de tamanho 3x3, e stride 1.
            nn.BatchNorm2d(8),# uma camada convolucional com 8 canais de entrada e saída, kernel de tamanho 3x3, e stride 1.
            nn.ReLU(),# função de ativação ReLU para introduzir não-linearidade.
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Value function - calculates the value of the state (calculates how good the state is, no matter the action)
        self.state_fc = nn.Sequential(
            nn.Linear(96, 64), #camada linear que reduz a entrada de 96 para 64 neurônios
            nn.ReLU(), # função de ativação ReLU
            nn.Linear(64, 1)# camada linear que reduz a entrada de 64 para 1 neurônio, representando o valor do estado.
            )

        # Advantage function - calculates the relative advantage of each action in the state
        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64), # camada linear que reduz a entrada de 96 para 64 neurônios
            nn.ReLU(), # função de ativação ReLU
            nn.Linear(64, available_actions_count) # camada linear que reduz a entrada de 64 para o número de ações disponíveis, representando a vantagem de cada ação.
        )

    def forward(self, x):
        x = self.conv1(x) # aplica a primeira camada convolucional
        x = self.conv2(x) # aplica a segunda camada convolucional
        x = self.conv3(x) # aplica a terceira camada convolucional
        x = self.conv4(x) # aplica a quarta camada convolucional

        x = x.view(-1, 192) # achata a saída da última camada convolucional para um vetor de tamanho 192

        x1 = x[:, :96]  # get the first 96 elements from the flat vector, to calculates the state value
        x2 = x[:, 96:]  # get the last 96 elements from the flat vector, to calculates the advantage values

        state_value = self.state_fc(x1).reshape(-1, 1) # calcula o valor do estado usando a rede neural

        advantage_values = self.advantage_fc(x2) # calcula a vantagem de cada ação usando a rede neural

        # Combina o valor do estado e as vantagens das ações para calcular os valores Q
        x = state_value + ( # soma o valor do estado com as vantagens das ações 
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1) # subtrai a média das vantagens das ações
        )

        return x # retorna os valores Q para cada ação


class DQNAgent:
    def __init__(
        self,
        action_size, # the number of possible actions
        memory_size, # the maximum size of the replay memory
        batch_size, # the number of experiences to sample from the replay memory
        discount_factor, # the discount factor for future rewards
        lr, # the learning rate
        load_model, # whether to load a pre-trained model
        epsilon=1,  # the initial exploration rate (epsilon-greedy strategy) - https://www.youtube.com/watch?v=e3L4VocZnnQ
        epsilon_decay=0.9996,
        epsilon_min=0.1,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile, weights_only=False)
            self.target_net = torch.load(model_savefile, weights_only=False)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr) # todo, aprimorar usar adam ou rmsprop

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min



if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size() # get number of buttons - docs/buttons_available.txt

    # Available actions: [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    # actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions = [[1,0,0], [0,1,0], [0,0,1]] # move left, move right, shoot


    # Initialize our agent with the set parameters
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    print("Initializing doom...")

    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
