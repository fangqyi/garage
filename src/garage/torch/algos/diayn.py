import torch
import torch.nn.functional as F
import numpy as np
from garage.torch.algos import SAC


class DIAYN(SAC):
    def __init__(self,
                 env_spec,
                 skills_num,
                 discriminator,
                 policy,
                 qf1,
                 qf2,
                 replay_buffer,
                 *,  # Everything after this is numbers
                 max_path_length,
                 max_eval_path_length=None,
                 gradient_steps_per_itr,
                 fixed_alpha=None,  # empirically could be 0.1
                 target_entropy=None,
                 initial_log_entropy=0.,
                 discount=0.99,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 target_update_tau=5e-3,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 reward_scale=1.0,
                 optimizer=torch.optim.Adam,
                 steps_per_epoch=1,
                 num_evaluation_trajectories=10,
                 eval_env=None):

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         replay_buffer=replay_buffer,
                         max_path_length=max_path_length,
                         max_eval_path_length=max_eval_path_length,
                         gradient_steps_per_itr=gradient_steps_per_itr,
                         fixed_alpha=fixed_alpha,
                         target_entropy=target_entropy,
                         initial_log_entropy=initial_log_entropy,
                         discount=discount,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         target_update_tau=target_update_tau,
                         policy_lr=policy_lr,
                         qf_lr=qf_lr,
                         reward_scale=reward_scale,
                         optimizer=optimizer,
                         steps_per_epoch=steps_per_epoch,
                         num_evaluation_trajectories=num_evaluation_trajectories,
                         eval_env=eval_env)

        self._skills_num = skills_num
        self._prob_skill = np.full(skills_num, 1.0/skills_num)
        self._discriminator = discriminator
        self._discriminator_optimizer=self._optimizer(self._discriminator.parameters(),
                                                      lr=self._policy_lr)


    def train(self, runner):
        if not self._eval_env:
            self._eval_env = runner.get_env_copy()
        last_return = None

        for _ in runner.step_epoches():
            for _ in range(self.steps_per_epoch):  # step refers to episode ?

                z = self._sample_skill()

                path_returns = []
                path = self._obtain_init_sample()
                for path in runner.step_path:
                    a = self._policy.

    def _obtain_init_sample(self):
        # TODO: Sample initial state s0~p0(s)
        path = dict()
        return path

    def _obtain_sample(self):
        path = dict()
        return path

    def _learn_once(self, samples_data):
        policy_loss, qf1_loss, qf2_loss = self.optimize_policy(0, samples_data)
        self._update_targets()
        discriminator_loss = self.optimize_discriminator(samples_data)

        return policy_loss, qf1_loss, qf2_loss, discriminator_loss

    # def _get_log_alpha(self, samples_data):
    #     pass

    # def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):

    # def _critic_objective(self, samples_data):

    def _discriminator_objective(self, samples_data):
        state = samples_data['next_state']
        skill = samples_data['skill']

        discriminator_pred = self._discriminator(state)
        discriminator_target = self._get_one_hot_tensor(self._skills_num, skill)

        discriminator_loss = F.mse_loss(discriminator_pred.flatten(), discriminator_target)

        return discriminator_loss

    # def _update_targets(self):

    def optimize_discriminator(self, samples_data):
        discriminator_loss = self._discriminator_objective(samples_data)

        self._discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self._discriminator_optimizer.step()

        return discriminator_loss

    # def optimize_policy(self, itr, samples_data):

    def _evaluate_policy(self, epoch):
        # TODO: test how it improves the training of sac

    # def _temperature_objective(self, log_pi, samples_data):
        pass

    @property
    def networks(self):
        return [
            self._policy, self._discriminator, self._qf1, self._qf2,
            self._target_qf1, self._target_qf2
        ]

    # def to(self, device=None):

    def _sample_skill(self):  # to maximize entropy
        return np.random.choice(self._skills_num, self._prob_skill)

    def _obtain_pseudo_reward(self, samples_data):
        state = samples_data['state']
        skill = samples_data['skill']
        q_z = self._discriminator(state)[skill]

        reward = np.log(q_z) - np.log(self._prob_skill)
        return torch.tensor(reward, dtype=torch.float)

    def _get_single_ndarry(self, x):
        return np.array([x])

    def _get_one_hot_tensor(self, size, idx):  # size-length vector
        input_onehot = np.zeros((1,size))
        input_onehot[:,idx] = 1.
        return torch.from_numpy(input_onehot)

